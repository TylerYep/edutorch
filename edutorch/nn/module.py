from typing import Any, Dict, Tuple

import numpy as np


class Module:
    def __init__(self) -> None:
        self.cache: Tuple[Any, ...] = ()
        self._params: Dict[str, Dict[str, Any]] = {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Modules must implement the forward pass.")

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError(
            "Modules must implement local backward pass to return parameter gradients."
        )

    def parameters(self) -> Dict[str, Dict[str, Any]]:
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
