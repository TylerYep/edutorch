from typing import Optional, Tuple

import numpy as np

from .functional import sigmoid
from .module import Module


class LSTMCell(Module):
    def __init__(
        self,
        prev_h: np.ndarray,
        prev_c: np.ndarray,
        Wx: np.ndarray,
        Wh: np.ndarray,
        b: np.ndarray,
    ) -> None:
        super().__init__()
        self.prev_h = prev_h
        self.prev_c = prev_c
        self.Wx = Wx
        self.Wh = Wh
        self.b = b
        self.next_h: Optional[np.array] = None
        self.next_c: Optional[np.array] = None
        self.set_parameters("prev_h", "prev_c", "Wx", "Wh", "b")

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        H = self.prev_h.shape[1]
        a = self.prev_h @ self.Wh + x @ self.Wx + self.b
        a_i, a_f, a_o, a_g = (
            a[:, :H],
            a[:, H : 2 * H],
            a[:, 2 * H : 3 * H],
            a[:, 3 * H :],
        )
        self.next_c = sigmoid(a_f) * self.prev_c + sigmoid(a_i) * np.tanh(a_g)
        self.next_h = sigmoid(a_o) * np.tanh(self.next_c)
        self.cache = (x, a)

        return self.next_h, self.next_c

    def backward(self, dout: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """

        def d_sigmoid(x: np.ndarray) -> np.ndarray:
            return sigmoid(x) * (1 - sigmoid(x))

        def d_tanh(x: np.ndarray) -> np.ndarray:
            return 1 - np.tanh(x) ** 2

        dnext_h, dnext_c = dout
        H = dnext_h.shape[1]
        x, a = self.cache
        a_i, a_f, a_o, a_g = (
            a[:, :H],
            a[:, H : 2 * H],
            a[:, 2 * H : 3 * H],
            a[:, 3 * H :],
        )
        i, f, o, g = sigmoid(a_i), sigmoid(a_f), sigmoid(a_o), np.tanh(a_g)

        dc = dnext_c + dnext_h * d_tanh(self.next_c) * o
        dprev_c = dc * f

        di = dc * g
        df = dc * self.prev_c
        do = dnext_h * np.tanh(self.next_c)
        dg = dc * i

        da_i = di * d_sigmoid(a_i)
        da_f = df * d_sigmoid(a_f)
        da_o = do * d_sigmoid(a_o)
        da_g = dg * d_tanh(a_g)

        da = np.concatenate((da_i, da_f, da_o, da_g), axis=1)

        dprev_h = da @ self.Wh.T
        dx = da @ self.Wx.T
        dWx = x.T @ da
        dWh = self.prev_h.T @ da
        db = np.sum(da, axis=0)

        return dx, dprev_h, dprev_c, dWx, dWh, db
