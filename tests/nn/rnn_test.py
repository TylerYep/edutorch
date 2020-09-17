import numpy as np

from edutorch.nn import RNN
from tests.gradient_check import estimate_gradients, rel_error


def test_rnn_forward() -> None:
    N, T, D, H = 2, 3, 4, 5
    x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
    model = RNN(D, H, N)
    model.h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
    model.Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
    model.Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
    model.b = np.linspace(-0.7, 0.1, num=H)

    h = model(x)

    expected_h = np.asarray(
        [
            [
                [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
                [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
                [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
            ],
            [
                [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
                [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
                [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043],
            ],
        ]
    )

    assert np.allclose(expected_h, h)


def test_rnn_backward() -> None:
    N, D, T, H = 2, 3, 10, 5

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    model = RNN(D, H, N)

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
