import numpy as np


def softmax(f: np.ndarray) -> np.ndarray:
    """ Numerically stable implementation. """
    f -= np.max(f)
    exp_f = np.exp(f)
    if len(f.shape) == 1:
        return exp_f / np.sum(exp_f, axis=0)
    return exp_f / np.sum(exp_f, axis=1).reshape(-1, 1)
