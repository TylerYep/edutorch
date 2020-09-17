from typing import Optional

import numpy as np

from .module import Module


class Dropout(Module):
    def __init__(
        self, p: float, train_mode: bool = True, seed: Optional[int] = None
    ) -> None:
        super().__init__()
        self.p = p
        self.train_mode = train_mode
        self.seed = seed

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
        - p: Dropout parameter. We keep each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
            if the mode is test, then just return the input.
        - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking but not
            in real networks.

        Outputs:
        - out: Array of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
        mask that was used to multiply the input; in test mode, mask is None.

        NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.

        NOTE: Keep in mind that p is the probability of **keep** a neuron
        output; this might be contrary to some sources, where it is referred to
        as the probability of dropping a neuron output.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.train_mode:
            mask = (np.random.rand(*x.shape) < self.p) / self.p
            out = x * mask
        else:
            mask = None
            out = x

        self.cache = (mask,)
        out = out.astype(x.dtype, copy=False)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from dropout_forward.
        """
        (mask,) = self.cache
        return dout * mask if self.train_mode else dout
