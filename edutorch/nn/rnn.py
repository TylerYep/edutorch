from typing import Tuple

import numpy as np

from .module import Module
from .rnn_cell import RNNCell


class RNN(Module):
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
        # Manually reset cache in order to correctly append to it
        self.cache = ()

        N, T, _ = x.shape
        H = self.hidden_size
        prev_h = self.h0

        h = np.zeros((N, T, H))
        for i in range(T):
            cell = RNNCell(prev_h, self.Wx, self.Wh, self.b)
            h[:, i, :] = cell(x[:, i, :])
            prev_h = h[:, i, :]
            self.cache += (cell,)
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
        N, T, _ = dout.shape
        D = self.input_size

        dx = np.zeros((N, T, D))
        dprev_h = np.zeros_like(self.h0)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)

        for i in reversed(range(T)):
            cell = self.cache[i]
            dx[:, i, :], dh_i, dWx_i, dWh_i, db_i = cell.backward(
                dout[:, i, :] + dprev_h
            )
            dWx += dWx_i
            dWh += dWh_i
            db += db_i
            dprev_h = dh_i

        dh0 = dprev_h
        return dx, dh0, dWx, dWh, db
