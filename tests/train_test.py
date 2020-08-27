from typing import Tuple

import numpy as np

from edutorch.losses import softmax_loss
from edutorch.nn import Linear, Module, ReLU
from edutorch.optim import Adam


class Example(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 30)
        self.relu = ReLU()
        self.fc2 = Linear(30, 10)
        self.set_parameters("fc1", "fc2")

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        grads = {}
        dx2, dw2, db2 = self.fc2.backward(dout)
        grads["fc2"] = {"w": dw2, "b": db2}
        dx3 = self.relu.backward(dx2)
        _, dw1, db1 = self.fc1.backward(dx3)
        grads["fc1"] = {"w": dw1, "b": db1}
        return grads


def test_training(fashion_mnist: Tuple[np.ndarray, ...]) -> None:
    X_train, y_train, _, _ = fashion_mnist

    model = Example()

    save1 = np.array(model.fc1.w)
    save2 = np.array(model.fc2.w)
    optimizer = Adam(model)

    for _ in range(100):
        out = model(X_train)
        loss, dx = softmax_loss(out, y_train)
        grads = model.backward(dx)
        optimizer.step(model, grads)

    assert loss < 1e-4
    assert not (model.fc1.w == save1).all()
    assert not (model.fc2.w == save2).all()
