import numpy as np
import pytest

from edutorch.nn import Linear, Module


class ExtraParameters(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)
        self.fc2 = Linear(30, 10)
        self.set_parameters("fc1", "fc3")

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        grads = {}
        dx2, dw2, db2 = self.fc2.backward(dout)
        grads["fc2"] = {"w": dw2, "b": db2}
        _, dw1, db1 = self.fc1.backward(dx2)
        grads["fc1"] = {"w": dw1, "b": db1}
        return grads


# pylint: disable=abstract-method
class MissingForwardBackward(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)


def test_extra_params() -> None:
    with pytest.raises(ValueError):
        _ = ExtraParameters()


def test_missing_forward_backward() -> None:
    model = MissingForwardBackward()
    with pytest.raises(NotImplementedError):
        model.forward(np.zeros([1]))

    with pytest.raises(NotImplementedError):
        model.backward(np.zeros([1]))
