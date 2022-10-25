import numpy as np


class PWrapper:
    """
    Provides wrapper for pymoo problems, since they require different calling convention than our implementation
    supports. Additionally, it provides vectorized version of the provided function 'vfunc', which can be used
    for plot generation.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return -1*float(self.func.evaluate(X=args[0])[0])

    def vfunc(self, *args):
        return np.vectorize(self.func.evaluate)(np.array(args))
