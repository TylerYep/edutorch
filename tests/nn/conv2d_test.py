import numpy as np

from edutorch.nn import Conv2d
from tests.gradient_check import estimate_gradients


def test_conv2d_forward() -> None:
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    model = Conv2d(3, 3, kernel_size=(4, 4), stride=2, pad=1)
    model.w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    model.b = np.linspace(-0.1, 0.2, num=3)
    out = model(x)

    correct_out = np.array(
        [
            [
                [[-0.08759809, -0.10987781], [-0.18387192, -0.2109216]],
                [[0.21027089, 0.21661097], [0.22847626, 0.23004637]],
                [[0.50813986, 0.54309974], [0.64082444, 0.67101435]],
            ],
            [
                [[-0.98053589, -1.03143541], [-1.19128892, -1.24695841]],
                [[0.69108355, 0.66880383], [0.59480972, 0.56776003]],
                [[2.36270298, 2.36904306], [2.38090835, 2.38247847]],
            ],
        ]
    )
    assert np.allclose(out, correct_out)


def test_conv2d_backward() -> None:
    x = np.random.randn(3, 3, 3, 3)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2)
    dout = np.random.randn(3, 2, 3, 3)

    model = Conv2d(1, 2, kernel_size=(4, 4), stride=1, pad=1)
    params = {"w": w, "b": b}
    dx_num, dw_num, db_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dw, db = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dw_num, dw)
    assert np.allclose(db_num, db)
