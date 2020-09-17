"""
This folder contains helper functions for common mathematical functions.
Note: this package is not equivalent to PyTorch's functional API, because
no backwards() methods are provided for these functions!

To use these functions as layers, please use the implementation in
layers/ or losses/ instead.
"""
import numpy as np


def softmax(f: np.ndarray) -> np.ndarray:
    """ Numerically stable implementation of softmax. """
    f -= np.max(f)
    exp_f = np.exp(f)
    if len(f.shape) == 1:
        return exp_f / np.sum(exp_f, axis=0)
    return exp_f / np.sum(exp_f, axis=1).reshape(-1, 1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """ A numerically stable version of the logistic sigmoid function. """
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
