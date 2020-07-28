import numpy as np

from edutorch.losses import temporal_softmax_loss
from tests.gradient_check import eval_numerical_gradient


def test_temporal_softmax_output() -> None:
    def check_loss(N: int, T: int, V: int, p: float) -> float:
        x = 0.001 * np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = np.random.rand(N, T) <= p
        return temporal_softmax_loss(x, y, mask)[0]

    assert abs(check_loss(100, 1, 10, 1.0) - 2.3) < 5e-2
    assert abs(check_loss(100, 10, 10, 1.0) - 23) < 5e-2
    assert abs(check_loss(5000, 10, 10, 0.1) - 2.3) < 5e-2


def test_temporal_softmax_gradient() -> None:
    N, T, V = 7, 8, 9
    x = np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = np.random.rand(N, T) > 0.5

    _, dx = temporal_softmax_loss(x, y, mask)
    dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x)

    assert np.allclose(dx, dx_num)
