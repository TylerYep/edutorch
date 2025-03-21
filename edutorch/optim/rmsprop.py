from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np

from .optimizer import Optimizer

if TYPE_CHECKING:
    from edutorch.nn.module import Module
    from edutorch.nptypes import NPArray


@dataclass
class RMSProp(Optimizer):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
        gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """

    model: Module
    lr: float = 1e-2
    decay_rate: float = 0.99
    eps: float = 1e-8

    @override
    def init_context(self, w: NPArray) -> tuple[Any, ...]:
        """Initialize context using weights."""
        v = np.zeros_like(w)
        return (v,)

    @override
    def update(
        self, context: tuple[Any, ...], w: NPArray, dw: NPArray
    ) -> tuple[NPArray, tuple[NPArray, ...]]:
        (v,) = context

        v = self.decay_rate * v + (1 - self.decay_rate) * dw**2
        w -= self.lr * dw / (np.sqrt(v) + self.eps)

        return w, (v,)
