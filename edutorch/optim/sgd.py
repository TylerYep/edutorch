from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from edutorch.nn.module import Module
from edutorch.optim.optimizer import Optimizer
from edutorch.typing import NPArray


@dataclass
class SGD(Optimizer):
    """
    Performs vanilla stochastic gradient descent.
    """

    model: Module
    lr: float = 1e-2

    @override
    def init_context(self, w: NPArray) -> tuple[Any, ...]:
        """Initialize context using weights."""
        del w
        return ()

    @override
    def update(
        self, context: tuple[Any, ...], w: NPArray, dw: NPArray
    ) -> tuple[NPArray, tuple[NPArray, ...]]:
        del context

        w -= self.lr * dw

        return w, ()
