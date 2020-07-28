import numpy as np

from edutorch.nn import MaxPool2d
from tests.gradient_check import estimate_gradients


def test_maxpool2d_forward() -> None:
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    model = MaxPool2d(kernel_size=(2, 2), stride=2)
    out = model(x)

    correct_out = np.array(
        [
            [
                [[-0.26315789, -0.24842105], [-0.20421053, -0.18947368]],
                [[-0.14526316, -0.13052632], [-0.08631579, -0.07157895]],
                [[-0.02736842, -0.01263158], [0.03157895, 0.04631579]],
            ],
            [
                [[0.09052632, 0.10526316], [0.14947368, 0.16421053]],
                [[0.20842105, 0.22315789], [0.26736842, 0.28210526]],
                [[0.32631579, 0.34105263], [0.38526316, 0.4]],
            ],
        ]
    )
    assert np.allclose(out, correct_out)


def test_maxpool2d_backward() -> None:
    x = np.random.randn(3, 2, 4, 4)
    dout = np.random.randn(3, 2, 2, 2)

    model = MaxPool2d(kernel_size=(2, 2), stride=2)
    dx_num = estimate_gradients(model, dout, x, {})

    _ = model(x)
    dx = model.backward(dout)

    assert np.allclose(dx_num, dx)
