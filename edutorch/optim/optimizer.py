from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edutorch.nn.module import Module
    from edutorch.nptypes import NPArray


@dataclass
class Optimizer:
    model: Module

    def __post_init__(self) -> None:
        self.context = self.set_context(self.model.parameters())

    def set_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Traverses down the model.parameters() tree and initializes all leaf
        nodes using the overridden init_context function.
        """
        for param_name, param in context.items():
            context[param_name] = (
                self.set_context(param)
                if isinstance(param, dict)
                else self.init_context(param)
            )
        return context

    def init_context(self, w: NPArray) -> tuple[Any, ...]:
        """This function initializes any context variables, running averages, etc."""
        raise NotImplementedError

    def update(
        self, context: tuple[Any, ...], w: NPArray, dw: NPArray
    ) -> tuple[NPArray, tuple[NPArray, ...]]:
        """
        This function updates a single weight using the context and values of w and dw.
        """
        raise NotImplementedError

    def step(self, model: Module, gradients: dict[str, Any]) -> None:
        """
        Model parameters and gradients should be matching dictionaries.
        Traverse both dictionaries simultaneously - on each branch,
        recurse on the branch.
        On a leaf, pass the context into the _step function.

        Parameter context should always be None from the user's perspective.
        """

        def _step(
            model: Module, gradients: dict[str, Any], context: dict[str, Any]
        ) -> None:
            for param_name, param in model.parameters().items():
                step_context = context[param_name]
                if param_name not in gradients:
                    raise ValueError(
                        f"{model.__class__.__name__} has no gradient for {param_name}. "
                        f"Please ensure {param_name} was assigned a gradient "
                        "in model.backward()."
                    )

                # On a branch, recurse on that branch
                if isinstance(param, dict):
                    submodel = getattr(model, param_name)
                    subgradients: dict[str, NPArray] = gradients[param_name]
                    _step(submodel, subgradients, step_context)

                # On a leaf, update the weight in that leaf
                else:
                    w, dw = getattr(model, param_name), gradients[param_name]  # type: ignore[unreachable]
                    if w.shape != dw.shape:
                        raise ValueError(
                            f"Shapes of w and dw do not match: {w.shape} {dw.shape}"
                        )
                    new_w, new_context = self.update(step_context, w, dw)
                    setattr(model, param_name, new_w)
                    context[param_name] = new_context

        _step(model, gradients, self.context)
