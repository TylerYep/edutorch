from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ..nn.module import Module
from .optimizer import Optimizer


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

    def init_context(self, w: np.ndarray) -> Tuple[Any, ...]:
        """ Initialize context using weights. """
        b = self.momentum
        v = np.zeros_like(w)
        return b, v

    def update(
        self, context: Tuple[Any, ...], w: np.ndarray, dw: np.ndarray
    ) -> np.ndarray:
        (b, v) = context

        v = b * v - self.lr * dw
        w += v

        return w, (b, v)
