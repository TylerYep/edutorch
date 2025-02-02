from typing import override

import numpy as np

from edutorch.nptypes import NPArray

from .module import Module


class ReLU(Module):
    @override
    def forward(self, x: NPArray) -> NPArray:
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        self.cache = (x,)
        return np.maximum(0, x)

    @override
    def backward(self, dout: NPArray) -> NPArray:
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        (x,) = self.cache
        return np.where(x > 0, dout, 0)
