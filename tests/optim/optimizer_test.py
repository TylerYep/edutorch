from typing import Tuple

import numpy as np
import pytest

from edutorch.losses import softmax_loss
from edutorch.nn import Linear, Module
from edutorch.optim import Adam


class MissingGradients(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)
        self.fc2 = Linear(30, 10)
        self.set_parameters("fc1", "fc2")

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx2, _, _ = self.fc2.backward(dout)
        _, _, _ = self.fc1.backward(dx2)
        return {}


class MissingParameters(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)
        self.fc2 = Linear(30, 10)

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


def test_missing_params(fashion_mnist: Tuple[np.ndarray, ...]) -> None:
    X_train, y_train, _, _ = fashion_mnist

    model = MissingParameters()

    save1 = np.array(model.fc1.w)
    save2 = np.array(model.fc2.w)
    optimizer = Adam(model)

    out = model(X_train)
    _, dx = softmax_loss(out, y_train)
    grads = model.backward(dx)
    optimizer.step(model, grads)

    assert (model.fc1.w == save1).all()
    assert (model.fc2.w == save2).all()


def test_missing_gradients(fashion_mnist: Tuple[np.ndarray, ...]) -> None:
    X_train, y_train, _, _ = fashion_mnist

    model = MissingGradients()
    optimizer = Adam(model)

    out = model(X_train)
    _, dx = softmax_loss(out, y_train)
    grads = model.backward(dx)

    with pytest.raises(ValueError):
        optimizer.step(model, grads)
