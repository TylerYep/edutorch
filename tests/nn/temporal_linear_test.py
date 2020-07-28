import numpy as np

from edutorch.nn import TemporalLinear
from tests.gradient_check import estimate_gradients


def test_temporal_linear_forward() -> None:
    pass


def test_temporal_linear_backward() -> None:
    N, T, D, M = 2, 3, 4, 5
    x = np.random.randn(N, T, D)
    w = np.random.randn(D, M)
    b = np.random.randn(M)
    dout = np.random.randn(N, T, M)

    model = TemporalLinear(D, M)
    params = {"w": w, "b": b}
    dx_num, dw_num, db_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dw, db = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dw_num, dw)
    assert np.allclose(db_num, db)
