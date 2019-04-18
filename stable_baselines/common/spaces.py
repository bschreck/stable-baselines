from gym.spaces import Discrete, Discrete, MultiDiscrete, MultiBinary, Box, Dict, Tuple
import numpy as np


class MixedMultiDiscreteBox(Tuple):
    def __init__(self, discrete_nvec, box_low, box_high, box_shape=None, dtype=np.float32):
        if len(box_shape) > 1:
            raise NotImplementedError("Boxes with len(shape) > 1 not yet implemented")
        super().__init__([MultiDiscrete(discrete_nvec),
                          Box(low=box_low, high=box_high, shape=box_shape, dtype=dtype)])
        self.shape = (self.spaces[0].shape[0] + self.spaces[1].shape[0],)
        self.dtype = self.spaces[1].dtype

    @property
    def multi_discrete(self):
        return self.spaces[0]

    @property
    def box(self):
        return self.spaces[1]

    def sample(self):
        tuple_sample = super().sample()
        return np.concatenate([tuple_sample[0].astype(self.spaces[1].dtype),
                               tuple_sample[1]], axis=-1)

    def __repr__(self):
        return "MixedMultiDiscreteBox(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def __eq__(self, other):
        return isinstance(other, MixedMultiDiscreteBox) and self.spaces == other.spaces

