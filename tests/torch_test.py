import numpy as np

from edutorch.layers.linear import Linear, Module
from edutorch.losses import softmax_loss
from edutorch.optimizers import Adam


class Example(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 30)
        self.fc2 = Linear(30, 10)
        self.set_parameters("fc1", "fc2")

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def backward(self, dout):
        grads = {}
        dx2, dw2, db2 = self.fc2.backward(dout)
        grads["fc2"] = {"w": dw2, "b": db2}
        _, dw1, db1 = self.fc1.backward(dx2)
        grads["fc1"] = {"w": dw1, "b": db1}
        return grads


def test_torch(fashion_mnist):
    np.random.seed(0)
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
        print(loss)

    assert not (model.fc1.w == save1).all()
    assert not (model.fc2.w == save2).all()
