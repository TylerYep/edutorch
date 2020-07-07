from typing import Tuple

import numpy as np

from .module import Module


class TemporalLinear(Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w = np.random.normal(scale=1e-3, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.set_parameters("w", "b")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for a temporal affine layer. The input is a set of D-dimensional
        vectors arranged into a minibatch of N timeseries, each of length T. We use
        an affine function to transform each of those vectors into a new vector of
        dimension M.

        Inputs:
        - x: Input data of shape (N, T, D)
        - w: Weights of shape (D, M)
        - b: Biases of shape (M,)

        Returns a tuple of:
        - out: Output data of shape (N, T, M)
        - cache: Values needed for the backward pass
        """
        N, T, D = x.shape
        M = self.b.shape[0]
        self.cache = (x,)
        return x.reshape(N * T, D).dot(self.w).reshape(N, T, M) + self.b

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Backward pass for temporal affine layer.

        Input:
        - dout: Upstream gradients of shape (N, T, M)
        - cache: Values from forward pass

        Returns a tuple of:
        - dx: Gradient of input, of shape (N, T, D)
        - dw: Gradient of weights, of shape (D, M)
        - db: Gradient of biases, of shape (M,)
        """
        (x,) = self.cache
        N, T, D = x.shape
        M = self.b.shape[0]

        dx = dout.reshape(N * T, M).dot(self.w.T).reshape(N, T, D)
        dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
        db = dout.sum(axis=(0, 1))

        return dx, dw, db
