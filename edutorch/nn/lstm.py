from typing import Tuple

import numpy as np

from .lstm_cell import LSTMCell
from .module import Module


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, batch_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        N, D, H = batch_size, input_size, hidden_size
        self.h0 = np.random.normal(scale=1e-3, size=(N, H))
        self.Wx = np.random.normal(scale=1e-3, size=(D, H))
        self.Wh = np.random.normal(scale=1e-3, size=(H, H))
        self.b = np.random.normal(scale=1e-3, size=H)
        self.set_parameters("h0", "Wx", "Wh", "b")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        # Manually reset cache in order to correctly append to it
        self.cache = ()

        N, T, _ = x.shape
        H = self.hidden_size
        prev_h = self.h0
        prev_c = np.zeros((N, H))

        h = np.zeros((N, T, H))
        for i in range(T):
            cell = LSTMCell(prev_h, prev_c, self.Wx, self.Wh, self.b)
            h[:, i, :], prev_c = cell(x[:, i, :])
            prev_h = h[:, i, :]
            self.cache += ((cell, prev_c),)

        return h

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dout/dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        N, T, _ = dout.shape
        D = self.input_size
        _, prev_c = self.cache[-1]

        dx = np.zeros((N, T, D))
        dprev_h = np.zeros_like(self.h0)
        dprev_c = np.zeros_like(prev_c)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)

        for i in reversed(range(T)):
            cell, _ = self.cache[i]
            dx[:, i, :], dh_i, dc_i, dWx_i, dWh_i, db_i = cell.backward(
                (dout[:, i, :] + dprev_h, dprev_c)
            )
            dWx += dWx_i
            dWh += dWh_i
            db += db_i
            dprev_h = dh_i
            dprev_c = dc_i

        dh0 = dprev_h
        return dx, dh0, dWx, dWh, db
