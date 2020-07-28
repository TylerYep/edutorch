import numpy as np

from edutorch.nn import Embedding
from tests.gradient_check import estimate_gradients


def test_embedding_forward() -> None:
    V, D = 5, 3

    x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
    model = Embedding(V, D)
    model.W = np.linspace(0, 1, num=V * D).reshape(V, D)

    out = model(x)

    expected_out = np.asarray(
        [
            [
                [0.0, 0.07142857, 0.14285714],
                [0.64285714, 0.71428571, 0.78571429],
                [0.21428571, 0.28571429, 0.35714286],
                [0.42857143, 0.5, 0.57142857],
            ],
            [
                [0.42857143, 0.5, 0.57142857],
                [0.21428571, 0.28571429, 0.35714286],
                [0.0, 0.07142857, 0.14285714],
                [0.64285714, 0.71428571, 0.78571429],
            ],
        ]
    )

    assert np.allclose(expected_out, out)


def test_embedding_backward() -> None:
    N, T, V, D = 50, 3, 5, 6
    x = np.random.randint(V, size=(N, T))
    W = np.random.randn(V, D)
    dout = np.random.randn(N, T, D)

    model = Embedding(V, D)

    params = {"W": W}
    _, dW_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dW = model.backward(dout)

    assert np.allclose(dW, dW_num)
