from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edutorch.typing import NPArray

from ..nn.module import Module
from .optimizer import Optimizer


@dataclass
class SGD(Optimizer):
    """
    Performs vanilla stochastic gradient descent.
    """

    model: Module
    lr: float = 1e-2

    def init_context(self, w: NPArray) -> tuple[Any, ...]:
        """Initialize context using weights."""
        del w
        return ()

    def update(
        self, context: tuple[Any, ...], w: NPArray, dw: NPArray
    ) -> tuple[NPArray, tuple[NPArray, ...]]:
        del context

        w -= self.lr * dw

        return w, ()
