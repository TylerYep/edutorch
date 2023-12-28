from __future__ import annotations

from typing import override

import numpy as np

from edutorch.typing import NPArray

from .module import Module


class RNNCell(Module):
    def __init__(self, prev_h: NPArray, Wx: NPArray, Wh: NPArray, b: NPArray) -> None:
        super().__init__()
        self.prev_h = prev_h
        self.Wx = Wx
        self.Wh = Wh
        self.b = b
        self.next_h: NPArray
        self.set_parameters("prev_h", "Wx", "Wh", "b")

    @override
    def forward(self, x: NPArray) -> NPArray:
        """
        Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
        activation function.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Inputs:
        - x: Input data for this timestep, of shape (N, D).
        - prev_h: Hidden state from previous timestep, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - cache: Tuple of values needed for the backward pass.
        """
        self.next_h = np.tanh(self.prev_h @ self.Wh + x @ self.Wx + self.b)
        self.cache = (x,)
        return self.next_h

    @override
    def backward(self, dout: NPArray) -> tuple[NPArray, ...]:
        """
        Backward pass for a single timestep of a vanilla RNN.

        Inputs:
        - dout/dnext_h: Gradient of loss w/ respect to next hidden state (N, H)
        - cache: Cache object from the forward pass

        Returns a tuple of:
        - dx: Gradients of input data, of shape (N, D)
        - dprev_h: Gradients of previous hidden state, of shape (N, H)
        - dWx: Gradients of input-to-hidden weights, of shape (D, H)
        - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        - db: Gradients of bias vector, of shape (H,)
        """
        if self.next_h is None:
            raise RuntimeError("self.next_h should not be None.")

        (x,) = self.cache
        dh = dout * (1 - self.next_h**2)
        dx = dh @ self.Wx.T
        dprev_h = dh @ self.Wh.T
        dWx = x.T @ dh
        dWh = self.prev_h.T @ dh
        db = np.sum(dh, axis=0)
        return dx, dprev_h, dWx, dWh, db
