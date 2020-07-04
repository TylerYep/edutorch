from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from ..layers.module import Module


@dataclass
class Optimizer:
    model: Module

    def __post_init__(self) -> None:
        self.context = self.model.parameters()
        self.set_context(self.context)

    def set_context(self, context: Any) -> None:
        """
        Traverses down the model.parameters() tree and initializes all leaf
        nodes using the overridden init_context function.
        """
        for param_name, param in context.items():
            if isinstance(param, dict):
                self.set_context(param)
            else:
                context[param_name] = self.init_context(param)

    def init_context(self, w: np.ndarray) -> Tuple[Any, ...]:
        """ This function initializes any context variables, running averages, etc. """
        raise NotImplementedError

    def update(self, context: Tuple[Any, ...], w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        """ This function updates a single weight using the context and values of w and dw. """
        raise NotImplementedError

    def step(self, model: Module, gradients: Dict[str, np.ndarray]) -> None:
        """
        Model parameters and gradients should be matching dictionaries.
        Traverse both dictionaries simultaneously - on each branch, recurse on the branch.
        On a leaf, pass the context into the _step function.

        Parameter context should always be None from the user's perspective.
        """

        def _step(model: Module, gradients: Dict[str, np.ndarray], context: Any) -> None:
            for param_name, param in model.parameters().items():
                step_context = context[param_name]
                submodel = getattr(model, param_name)
                grad = gradients[param_name]

                # On a branch, recurse on that branch
                if isinstance(param, dict):
                    _step(submodel, grad, step_context)

                # On a leaf, update the weight in that leaf
                elif hasattr(model, param_name):
                    w, dw = submodel, grad
                    if w.shape != dw.shape:
                        raise ValueError(f"Shapes of w and dw do not match: {w.shape} {dw.shape}")
                    new_w, new_context = self.update(step_context, w, dw)
                    setattr(model, param_name, new_w)
                    context[param_name] = new_context

                else:
                    raise ValueError(f"{model.__class__.__name__} has no attribute: {param_name}")

        _step(model, gradients, self.context)
