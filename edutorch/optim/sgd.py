from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ..nn.module import Module
from .optimizer import Optimizer


@dataclass
class SGD(Optimizer):
    """
    Performs vanilla stochastic gradient descent.
    """

    model: Module
    lr: float = 1e-2

    def init_context(self, w: np.ndarray) -> Tuple[Any, ...]:
        """ Initialize context using weights. """
        del w
        return ()

    def update(
        self, context: Tuple[Any, ...], w: np.ndarray, dw: np.ndarray
    ) -> np.ndarray:
        del context

        w -= self.lr * dw

        return w
