from __future__ import annotations

from typing import Any

from edutorch.typing import NPArray


class Module:
    def __init__(self) -> None:
        self.cache: tuple[Any, ...] = ()
        self._params: dict[str, dict[str, Any]] = {}

    def __call__(self, x: Any) -> NPArray:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        raise NotImplementedError("Modules must implement the forward pass.")

    def backward(self, dout: Any) -> Any:
        raise NotImplementedError(
            "Modules must implement local backward pass to return parameter gradients."
        )

    def parameters(self) -> dict[str, dict[str, Any]]:
        """
        This function returns the parameters of the model that need gradients,
        in the order that they are returned in backward().
        """
        return self._params

    def set_parameters(self, *layers: str) -> None:
        for layer_name in layers:
            if hasattr(self, layer_name):
                submodel = getattr(self, layer_name)
                self._params[layer_name] = (
                    submodel.parameters() if isinstance(submodel, Module) else submodel
                )
            else:
                raise ValueError(
                    f"Not valid attribute of {self.__class__.__name__}: {layer_name}"
                )
