from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
import pytest

from edutorch.nn import Linear, Module

if TYPE_CHECKING:
    from edutorch.nptypes import NPArray


class ExtraParameters(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)
        self.fc2 = Linear(30, 10)
        self.set_parameters("fc1", "fc3")

    @override
    def forward(self, x: NPArray) -> NPArray:
        x = self.fc1(x)
        return self.fc2(x)

    @override
    def backward(self, dout: NPArray) -> dict[str, dict[str, NPArray]]:
        grads = {}
        dx2, dw2, db2 = self.fc2.backward(dout)
        grads["fc2"] = {"w": dw2, "b": db2}
        _, dw1, db1 = self.fc1.backward(dx2)
        grads["fc1"] = {"w": dw1, "b": db1}
        return grads


class MissingForwardBackward(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)


def test_extra_params() -> None:
    with pytest.raises(ValueError, match="Not valid attribute of ExtraParameters: fc3"):
        _ = ExtraParameters()


def test_missing_forward_backward() -> None:
    model = MissingForwardBackward()
    with pytest.raises(NotImplementedError):
        model.forward(np.zeros([1]))

    with pytest.raises(NotImplementedError):
        model.backward(np.zeros([1]))
