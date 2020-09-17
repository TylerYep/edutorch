import numpy as np

from edutorch.nn import Linear
from tests.gradient_check import estimate_gradients


def test_linear_forward() -> None:
    input_dim = 2
    input_shape = (4, 5, 6)
    output_dim = 3
    input_size = input_dim * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(input_dim, *input_shape)
    model = Linear(input_dim, output_dim)
    model.w = np.linspace(-0.2, 0.3, num=weight_size).reshape(
        np.prod(input_shape), output_dim
    )
    model.b = np.linspace(-0.3, 0.1, num=output_dim)

    out = model(x)
    correct_out = np.array(
        [[1.49834967, 1.70660132, 1.91485297], [3.25553199, 3.5141327, 3.77273342]]
    )

    assert np.allclose(out, correct_out)


def test_linear_backward() -> None:
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    model = Linear(10, 5)

    params = {"w": w, "b": b}
    dx_num, dw_num, db_num = estimate_gradients(model, dout, x, params)
    _ = model(x)
    dx, dw, db = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dw_num, dw)
    assert np.allclose(db_num, db)
