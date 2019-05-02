import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

from stable_baselines.a2c.utils import linear
from stable_baselines.common import spaces


class ProbabilityDistribution(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilites
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability

        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leiber divergence from the given probabilty distribution

        :param other: ([float]) the distibution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probabilty distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """

    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        Returns the probability distribution from flat probabilities
        flat: flattened vector of parameters of probability distribution

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param pi_latent_vector: ([float]) the latent pi values
        :param vf_latent_vector: ([float]) the latent vf values
        :param init_scale: (float) the inital scale of the distribution
        :param init_bias: (float) the inital bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        raise NotImplementedError

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class CategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_vec, scope=None):
        """
        The probability distribution type for multiple categorical input

        :param n_vec: ([int]) the vectors
        """
        self.scope = scope or ''
        # Cast the variable because tf does not allow uint32
        self.n_vec = n_vec.astype(np.int32)
        # Check that the cast was valid
        assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"

    def probability_distribution_class(self):
        return MultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiCategoricalProbabilityDistribution(self.n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(self.scope):
            pdparam = linear(pi_latent_vector, 'pi', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
            q_values = linear(vf_latent_vector, 'q', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [sum(self.n_vec)]

    def sample_shape(self):
        return [len(self.n_vec)]

    def sample_dtype(self):
        return tf.int32


class MultiMixedProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, categorical_n_vec, gaussian_size, scope=None):
        """
        The probability distribution type for multiple categorical and gaussian inputs

        :param categorical_n_vec: ([int]) the vectors for categorical inputs
        :param gaussian_size: (int) the number of dimensions of the multivariate gaussian
        """
        self.scope = scope or ''
        self.categorical_n_vec = categorical_n_vec
        self.gaussian_size = gaussian_size
        #self.multi_cat = MultiCategoricalProbabilityDistributionType(self.categorical_n_vec, scope='MultiCategoricalDist')
        self.gaussian = DiagGaussianProbabilityDistributionType(self.gaussian_size, scope='DiagGaussianDist')

    def probability_distribution_class(self):
        return MultiMixedProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiMixedProbabilityDistribution(self.categorical_n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(self.scope):
            mean = linear(pi_latent_vector, 'gaussian/pi', self.gaussian_size, init_scale=init_scale, init_bias=init_bias)
            logstd = tf.get_variable(name='gaussian/pi/logstd', shape=[1, self.gaussian_size], initializer=tf.zeros_initializer())
            # gauss_pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            gauss_q_values = linear(vf_latent_vector, 'gaussian/q', self.gaussian_size, init_scale=init_scale, init_bias=init_bias)

            # cat_pdparam = linear(pi_latent_vector, 'categorical/pi', sum(self.categorical_n_vec), init_scale=init_scale, init_bias=init_bias)
            #cat_pdparam = tf.get_variable(name='fake_cat', shape=[1, sum(self.categorical_n_vec),])
            gauss_pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            # pdparam = tf.concat([mean*0.0 + cat_pdparam, mean, mean * 0.0 + logstd], axis=1)
            # q_values = linear(vf_latent_vector, 'q', sum(self.categorical_n_vec) + self.gaussian_size, init_scale=init_scale, init_bias=init_bias)
        #return self.proba_distribution_from_flat(pdparam), tf.concat([cat_pdparam, mean], axis=1), q_values
        #return self.proba_distribution_from_flat(pdparam),  mean, gauss_q_values
        return self.proba_distribution_from_flat(gauss_pdparam),  mean, gauss_q_values

    def param_shape(self):
        #return [self.multi_cat.param_shape()[0] + self.gaussian.param_shape()[0]]
        return [self.gaussian.param_shape()[0]]

    def sample_shape(self):
        #return [self.multi_cat.sample_shape()[0] + self.gaussian.sample_shape()[0]]
        return [2 + self.gaussian.sample_shape()[0]]
        # return self.gaussian.sample_shape()

    def sample_dtype(self):
        return tf.float32

class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size, scope=None):
        """
        The probability distribution type for multivariate gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.scope = scope or ''
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(self.scope):
            mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
            logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for bernoulli input

        :param size: (int) the number of dimensions of the bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return BernoulliProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=tf.stop_gradient(one_hot_actions))

    def kl(self, other):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (a_0 - tf.log(z_0) - a_1 + tf.log(z_1)), axis=-1)

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class MultiMixedProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, categorical_nvec=None, flat=None, categoricals=None, gaussian=None):
        """
        Probability distributions from mixed multicategorical gaussian inputs

        :param categorical_nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) part categorical logits input, part multivariate gaussian input data
        :param categoricals:  TODO
        :param gaussian: TODO

        """
        if categoricals is None:
            if categorical_nvec is None or flat is None:
                raise ValueError("Must provide either `nvec` and `flat`, or already instantiated distributions"\
                                 "`categoricals` and `gaussians`")
            self.cat_size_one_hot = sum(categorical_nvec)
            self.cat_size = len(categorical_nvec)

            # self.categoricals = MultiCategoricalProbabilityDistribution(categorical_nvec,
                                                                        # self._get_cat_part_one_hot(flat))
            # self.gaussian = DiagGaussianProbabilityDistribution(self._get_gauss_part_one_hot(flat))
            self.gaussian = DiagGaussianProbabilityDistribution(flat)
            self.flat = flat
        else:
            self.categoricals = categoricals
            self.gaussian = gaussian
            self.cat_size = sum([cat.logits.get_shape()[-1].value for cat in self.categoricals.categoricals])
            # self.cat_size = len(self.categoricals.categoricals)#.flat.get_shape()[-1].value
        # prob_a_dist_cat, pdparam_cat, q_values_cat = self.multi_cat.proba_distribution_from_latent(
            # pi_latent_vector, vf_latent_vector, init_scale=init_scale, init_bias=init_bias)
        # prob_a_dist_gauss, pdparam_gauss, q_values_gauss = self.gaussian.proba_distribution_from_latent(
            # pi_latent_vector, vf_latent_vector, init_scale=init_scale, init_bias=init_bias)
        # return (MultiMixedProbabilityDistribution(categoricals=prob_a_dist_cat, gaussian=prob_a_dist_gauss),
                # tf.concat([pdparam_cat, pdparam_gauss], axis=-1),
                # tf.concat([q_values_cat, q_values_gauss], axis=-1))



    def _get_cat_part_one_hot(self, _input):
        return _input[:, :self.cat_size_one_hot]

    def _get_cat_part(self, _input):
        return _input[:, :self.cat_size]

    def _get_gauss_part_one_hot(self, _input):
        return _input[:, self.cat_size_one_hot:]

    def _get_gauss_part(self, _input):
        return _input[:, self.cat_size:]

    def flatparam(self):
        return self.flat

    def mode(self):
        #cat_mode = tf.cast(self.categoricals.mode(), tf.float32)
        gauss_mode = self.gaussian.mode()
        #return tf.concat([cat_mode, gauss_mode], axis=-1)
        return gauss_mode

    def neglogp(self, x):
        # TODO: this cast?
        #cat_x = tf.cast(self._get_cat_part(x), tf.int64)
        gauss_x = self._get_gauss_part(x)
        #return tf.add_n([self.categoricals.neglogp(cat_x), self.gaussian.neglogp(gauss_x)])
        # gauss_x = x
        return self.gaussian.neglogp(gauss_x)

    def kl(self, other):
        #cat_kl = self.categoricals.kl(other.categoricals)
        gauss_kl = self.gaussian.kl(other.gaussian)
        #return tf.add_n([cat_kl, gauss_kl])
        return gauss_kl

    def entropy(self):
        #cat_entropy = self.categoricals.entropy()
        gauss_entropy = self.gaussian.entropy()
        #return tf.add_n([cat_entropy, gauss_entropy])
        return gauss_entropy

    def sample(self):
        gauss_sample = self.gaussian.sample()
        #cat_sample = tf.cast(self.categoricals.sample(), tf.float32)
        # TODO: here
        with tf.variable_scope("categoricals", reuse=tf.AUTO_REUSE):
            cat_sample = tf.get_variable('cat_sample',
                                     shape=[1, self.cat_size],
                                     trainable=False,
                                     initializer=tf.zeros_initializer())
            cat_sample = gauss_sample * 0.0 + cat_sample
        return tf.concat([cat_sample, gauss_sample], axis=-1)
        #return gauss_sample

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Bounds are taken into acount outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate gaussian input data
        """
        return cls(flat)


class BernoulliProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from bernoulli input

        :param logits: ([float]) the bernoulli input data
        """
        self.logits = logits
        self.probabilities = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.probabilities)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=tf.cast(x, tf.float32)),
                             axis=-1)

    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits,
                                                                     labels=self.probabilities), axis=-1) - \
               tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def sample(self):
        samples_from_uniform = tf.random_uniform(tf.shape(self.probabilities))
        return tf.cast(math_ops.less(samples_from_uniform, self.probabilities), tf.float32)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new bernoulli input

        :param flat: ([float]) the bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given bernoulli input data
        """
        return cls(flat)


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the approriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MixedMultiDiscreteBox):
        return MultiMixedProbabilityDistributionType(ac_space.multi_discrete.nvec,
                                                     ac_space.box.shape[0])
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")


def shape_el(tensor, index):
    """
    get the shape of a TensorFlow Tensor element

    :param tensor: (TensorFlow Tensor) the input tensor
    :param index: (int) the element
    :return: ([int]) the shape
    """
    maybe = tensor.get_shape()[index]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(tensor)[index]
