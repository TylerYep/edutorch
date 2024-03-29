from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np

from .optimizer import Optimizer

if TYPE_CHECKING:
    from edutorch.nn.module import Module
    from edutorch.typing import NPArray


@dataclass
class SGDMomentum(Optimizer):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
        Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
        moving average of the gradients.
    """

    model: Module
    lr: float = 1e-2
    momentum: float = 0.9

    @override
    def init_context(self, w: NPArray) -> tuple[Any, ...]:
        """Initialize context using weights."""
        b = self.momentum
        v = np.zeros_like(w)
        return b, v

    @override
    def update(
        self, context: tuple[Any, ...], w: NPArray, dw: NPArray
    ) -> tuple[NPArray, tuple[NPArray, ...]]:
        (b, v) = context

        v = b * v - self.lr * dw
        w += v

        return w, (b, v)
