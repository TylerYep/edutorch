import numpy as np

from edutorch.nn import ReLU
from tests.gradient_check import estimate_gradients


def test_relu_forward() -> None:
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    model = ReLU()
    out = model(x)
    correct_out = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.04545455, 0.13636364],
            [0.22727273, 0.31818182, 0.40909091, 0.5],
        ]
    )

    assert np.allclose(out, correct_out)


def test_relu_backward() -> None:
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    model = ReLU()
    dx_num = estimate_gradients(model, dout, x, {})

    _ = model(x)
    dx = model.backward(dout)

    assert np.allclose(dx_num, dx)
