from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from .module import Module

if TYPE_CHECKING:
    from edutorch.nptypes import NPArray


class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w = np.random.normal(scale=1e-3, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.set_parameters("w", "b")

    @override
    def forward(self, x: NPArray) -> NPArray:
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
        We multiply this against a weight matrix of shape (D, M) where
        D = prod_i d_i

        Inputs:
        x - Input data, of shape (N, d_1, ..., d_k)
        w - Weights, of shape (D, M)
        b - Biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        self.cache = (x,)
        return x.reshape(x.shape[0], -1).dot(self.w) + self.b

    @override
    def backward(self, dout: NPArray) -> tuple[NPArray, ...]:
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        (x,) = self.cache
        dx = dout.dot(self.w.T).reshape(x.shape)
        dw = x.reshape(x.shape[0], -1).T.dot(dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db
