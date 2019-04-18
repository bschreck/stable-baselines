import numpy as np
from abc import ABC, abstractmethod
#from stable_baselines.common.spaces import MixedMultiDiscreteBox


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, n_steps):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        n_env = env.num_envs

        # if isinstance(env.observation_space, MixedMultiDiscreteBox):
            # observation_space_shape = [(len(env.observation_space[0].nvec),),
                                       # (env.observation_space[1].shape[0],)]
            # dtype_names = [ob_space.dtype.name for ob_space in env.observation_space]
            # self.batch_ob_shape = [(n_env*n_steps,) + ob_shape for ob_shape in observation_space_shape]
            # self.obs = [np.zeros((n_env,) + ob_space_shape, dtype=dtype_name)
                        # for ob_space_shape, dtype_name in zip(observation_space_shape,
                                                              # dtype_names)]
            # for i, o in enumerate(env.reset()):
                # self.obs[i][:] = o
        # else:
        self.batch_ob_shape = (n_env*n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        #JK:wkimport pdb; pdb.set_trace()
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_env)]

    @abstractmethod
    def run(self):
        """
        Run a learning step of the model
        """
        raise NotImplementedError
