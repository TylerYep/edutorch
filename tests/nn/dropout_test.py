import numpy as np

from edutorch.nn import Dropout
from tests.gradient_check import estimate_gradients


def test_dropout_forward() -> None:
    x = np.random.randn(500, 500) + 10

    for p in [0.25, 0.4, 0.7]:
        model = Dropout(p)
        out = model(x)
        model.train_mode = False
        out_test = model(x)

        assert np.allclose(x.mean(), 10, rtol=1e-2), "Mean of input"
        assert np.allclose(out.mean(), 10, rtol=1e-2), "Mean of train-time output"
        assert np.allclose(out_test.mean(), 10, rtol=1e-2), "Mean of test-time output"
        assert np.allclose(
            (out == 0).mean(), 1 - p, rtol=1e-2
        ), "Train-time frac set to 0"
        assert np.allclose(
            (out_test == 0).mean(), 0, rtol=1e-2
        ), "Test-time frac set to 0"


def test_dropout_backward() -> None:
    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)

    model = Dropout(p=0.2, seed=123)

    _ = model(x)
    dx = model.backward(dout)
    dx_num = estimate_gradients(model, dout, x, {})

    assert np.allclose(dx, dx_num)
