import numpy as np

from edutorch.nn import RNNCell
from tests.gradient_check import estimate_gradients


def test_rnn_cell_forward() -> None:
    N, D, H = 3, 10, 4
    x = np.linspace(-0.4, 0.7, num=N * D).reshape(N, D)

    model = RNNCell(
        prev_h=np.linspace(-0.2, 0.5, num=N * H).reshape(N, H),
        Wx=np.linspace(-0.1, 0.9, num=D * H).reshape(D, H),
        Wh=np.linspace(-0.3, 0.7, num=H * H).reshape(H, H),
        b=np.linspace(-0.2, 0.4, num=H),
    )

    next_h = model(x)
    expected_next_h = np.asarray(
        [
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [0.66854692, 0.79562378, 0.87755553, 0.92795967],
            [0.97934501, 0.99144213, 0.99646691, 0.99854353],
        ]
    )

    assert np.allclose(expected_next_h, next_h)


def test_rnn_cell_backward() -> None:
    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    prev_h = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)
    dnext_h = np.random.randn(*prev_h.shape)

    model = RNNCell(prev_h=prev_h, Wx=Wx, Wh=Wh, b=b)

    params = {"prev_h": prev_h, "Wx": Wx, "Wh": Wh, "b": b}
    dx_num, dprev_h_num, dWx_num, dWh_num, db_num = estimate_gradients(
        model, dnext_h, x, params
    )

    _ = model(x)
    dx, dprev_h, dWx, dWh, db = model.backward(dnext_h)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dprev_h_num, dprev_h)
    assert np.allclose(dWx_num, dWx)
    assert np.allclose(dWh_num, dWh)
    assert np.allclose(db_num, db)
