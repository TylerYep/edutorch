import numpy as np

from edutorch.nn import LSTM
from tests.gradient_check import estimate_gradients, rel_error


def test_lstm_forward() -> None:
    N, D, H, T = 2, 5, 4, 3
    x = np.linspace(-0.4, 0.6, num=N * T * D).reshape(N, T, D)
    model = LSTM(D, H, N)
    model.h0 = np.linspace(-0.4, 0.8, num=N * H).reshape(N, H)
    model.Wx = np.linspace(-0.2, 0.9, num=4 * D * H).reshape(D, 4 * H)
    model.Wh = np.linspace(-0.3, 0.6, num=4 * H * H).reshape(H, 4 * H)
    model.b = np.linspace(0.2, 0.7, num=4 * H)

    h = model(x)

    expected_h = np.asarray(
        [
            [
                [0.01764008, 0.01823233, 0.01882671, 0.0194232],
                [0.11287491, 0.12146228, 0.13018446, 0.13902939],
                [0.31358768, 0.33338627, 0.35304453, 0.37250975],
            ],
            [
                [0.45767879, 0.4761092, 0.4936887, 0.51041945],
                [0.6704845, 0.69350089, 0.71486014, 0.7346449],
                [0.81733511, 0.83677871, 0.85403753, 0.86935314],
            ],
        ]
    )

    assert np.allclose(expected_h, h)


def test_lstm_backward() -> None:
    N, D, T, H = 2, 3, 10, 6

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, 4 * H)
    Wh = np.random.randn(H, 4 * H)
    b = np.random.randn(4 * H)

    model = LSTM(D, H, N)

    model.h0 = h0
    model.Wx = Wx
    model.Wh = Wh
    model.b = b

    out = model(x)
    dout = np.random.randn(*out.shape)
    dx, dh0, dWx, dWh, db = model.backward(dout)

    params = {"h0": h0, "Wx": Wx, "Wh": Wh, "b": b}
    dx_num, dh0_num, dWx_num, dWh_num, db_num = estimate_gradients(
        model, dout, x, params
    )

    assert np.allclose(dx_num, dx), f"dx error: {rel_error(dx_num, dx)}"
    assert np.allclose(dh0_num, dh0), f"dh0 error: {rel_error(dh0_num, dh0)}"
    assert np.allclose(dWx_num, dWx), f"dWx error: {rel_error(dWx_num, dWx)}"
    assert np.allclose(dWh_num, dWh), f"dWh error: {rel_error(dWh_num, dWh)}"
    assert np.allclose(db_num, db), f"db error: {rel_error(db_num, db)}"
