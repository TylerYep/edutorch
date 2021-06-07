from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    def init_context(self, w: np.ndarray) -> tuple[Any, ...]:
        """Initialize context using weights."""
        del w
        return ()

    def update(
        self, context: tuple[Any, ...], w: np.ndarray, dw: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        del context

        w -= self.lr * dw

        return w, ()
