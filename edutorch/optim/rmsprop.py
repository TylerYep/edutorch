from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ..nn.module import Module
from .optimizer import Optimizer


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

    def init_context(self, w: np.ndarray) -> Tuple[Any, ...]:
        """ Initialize context using weights. """
        v = np.zeros_like(w)
        return (v,)

    def update(
        self, context: Tuple[Any, ...], w: np.ndarray, dw: np.ndarray
    ) -> np.ndarray:
        (v,) = context

        v = self.decay_rate * v + (1 - self.decay_rate) * dw ** 2
        w -= self.lr * dw / (np.sqrt(v) + self.eps)

        return w, (v,)
