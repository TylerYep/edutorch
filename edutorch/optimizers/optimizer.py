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

    def init_context(self, w: np.ndarray) -> Tuple[Any, ...]:  # pylint: disable=no-self-use
        """ Initialize context using weights. """
        del w
        return ()

    def set_context(self, context: Any) -> None:
        for param_name, param in context.items():
            if isinstance(param, dict):
                self.set_context(param)
            else:
                context[param_name] = self.init_context(param)

    def update(self, context: Tuple[Any, ...], w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, model: Module, gradients: Dict[str, np.ndarray]) -> None:
        """
        Model parameters and gradients should be matching dictionaries.
        Traverse both dictionaries - on a branch, recurse on the branch.
        On a leaf, pass the context into the _step function.

        Parameter context should always be None from the user's perspective.
        """

        def _step(model: Module, gradients: Dict[str, np.ndarray], context: Any) -> None:
            for param_name, param in model.parameters().items():
                submodel = getattr(model, param_name)  # Either a submodel or a weight w
                step_context = context[param_name]
                grad = gradients[param_name]
                if isinstance(param, dict):
                    _step(submodel, grad, step_context)
                elif hasattr(model, param_name):
                    # assert w.shape == dw.shape
                    assert submodel.shape == grad.shape
                    new_w, new_context = self.update(step_context, submodel, grad)
                    setattr(model, param_name, new_w)
                    context[param_name] = new_context
                else:
                    raise ValueError(f"{model.__class__.__name__} has no attribute: {param_name}")

        _step(model, gradients, self.context)
