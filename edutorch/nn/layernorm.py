from typing import Tuple

import numpy as np

from .module import Module


class LayerNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.set_parameters("gamma", "beta")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for layer normalization.

        During both training and test-time, the incoming data is normalized
        per data-point, before being scaled by gamma and beta parameters identical
        to that of batch normalization.

        Note that in contrast to batch normalization, the behavior during train
        and test-time for layer normalization are identical, and we do not need
        to keep track of running averages of any sort.

        Uses transpose matrix transformations to reuse batch norm code.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - ln_param: Dictionary with the following keys:
            - eps: Constant for numeric stability

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        x = x.T
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        v = np.sqrt(sample_var + self.eps)
        x_hat = (x - sample_mean) / v
        x_hat = x_hat.T
        out = self.gamma * x_hat + self.beta

        self.cache = (x_hat, v)

        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Backward pass for layer normalization.

        For this implementation, you can heavily rely on the work you've done already
        for batch normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from layernorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
        """
        x_hat, v = self.cache

        dx_hat = dout * self.gamma
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)

        x_hat = x_hat.T
        dx_hat = dx_hat.T
        dx = (
            dx_hat - np.mean(dx_hat, axis=0) - x_hat * np.mean(dx_hat * x_hat, axis=0)
        ) / v
        dx = dx.T

        return dx, dgamma, dbeta
