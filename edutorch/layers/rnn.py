from typing import Tuple

import numpy as np

from .module import Module
from .rnn_cell import RNNCell


# TODO The problem is the RNN cell cache is being overwritten at every timestep.


class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.h0 = None
        self.Wx = None
        self.Wh = None # np.random.normal(scale=1e-3, size=(hidden_size, hidden_size))
        self.b = None # np.random.normal(scale=1e-3, size=hidden_size)

        self.cell = None
        self.set_parameters("h0", "Wx", "Wh", "b")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run a vanilla RNN forward on an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The RNN uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the RNN forward, we return the hidden states for all timesteps.

        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D).
        - h0: Initial hidden state, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - h: Hidden states for the entire timeseries, of shape (N, T, H).
        - cache: Values needed in the backward pass
        """
        N, T, D = x.shape
        H = self.hidden_size
        if self.h0 is None:
            self.h0 = np.random.normal(scale=1e-3, size=(N, H))
            self.Wx = np.random.normal(scale=1e-3, size=(D, H))

        self.cell = RNNCell(self.h0, self.Wx, self.Wh, self.b)

        h = np.zeros((N, T, H))
        for i in range(T):
            h[:, i, :] = self.cell(x[:, i, :])
            self.cache += self.cell.cache
            self.cell.prev_h = self.cell.next_h
        return h

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute the backward pass for a vanilla RNN over an entire sequence of data.

        Inputs:
        - dout/dh: Upstream gradients of all hidden states, of shape (N, T, H).

        NOTE: 'dh' contains the upstream gradients produced by the
        individual loss functions at each timestep, *not* the gradients
        being passed between timesteps (which you'll have to compute yourself
        by calling rnn_step_backward in a loop).

        Returns a tuple of:
        - dx: Gradient of inputs, of shape (N, T, D)
        - dh0: Gradient of initial hidden state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        - db: Gradient of biases, of shape (H,)
        """
        N, T, H = dout.shape
        x = self.cache[-1]
        D = x.shape[1]

        dx = np.zeros((N, T, D))
        dprev_h = np.zeros_like(self.h0)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)

        for i in reversed(range(T)):
            self.cell.cache = self.cache[i]
            dx[:, i, :], dh_i, dWx_i, dWh_i, db_i = self.cell.backward(dout[:, i, :] + dprev_h)
            dWx += dWx_i
            dWh += dWh_i
            db += db_i
            dprev_h = dh_i

        dh0 = dprev_h
        return dx, dh0, dWx, dWh, db
