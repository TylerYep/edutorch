from typing import Tuple

import numpy as np

from .module import Module


class SpatialGroupNorm(Module):
    def __init__(self, num_features: int, G: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.G = G

        self.gamma = np.zeros(num_features)
        self.beta = np.zeros(num_features)
        self.set_parameters("gamma", "beta")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for spatial group normalization.
        In contrast to layer normalization, group normalization splits each entry
        in the data into G contiguous pieces, which it then normalizes independently.
        Per feature shifting and scaling are then applied to the data, in a manner
        identical to that of batch normalization and layer normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - G: Integer mumber of groups to split into, should be a divisor of C
        - gn_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        N, C, H, W = x.shape
        self.gamma = self.gamma.reshape((1, C, 1, 1))
        self.beta = self.beta.reshape((1, C, 1, 1))

        x = x.reshape(N * self.G, -1).T
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        v = np.sqrt(sample_var + self.eps)
        x_hat = (x - sample_mean) / v
        x_hat = x_hat.T.reshape(N, C, H, W)
        out = self.gamma * x_hat + self.beta

        self.cache = (x_hat, v)

        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Computes the backward pass for spatial group normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        x_hat, v = self.cache

        N, C, H, W = dout.shape
        dx_hat = dout * self.gamma
        dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        x_hat = x_hat.reshape(N * self.G, -1).T
        dx_hat = dx_hat.reshape(N * self.G, -1).T
        dx = (
            dx_hat - np.mean(dx_hat, axis=0) - x_hat * np.mean(dx_hat * x_hat, axis=0)
        ) / v
        dx = dx.T.reshape(N, C, H, W)

        return dx, dgamma, dbeta
