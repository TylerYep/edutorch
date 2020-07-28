import numpy as np

from edutorch.losses import softmax_loss
from tests.gradient_check import eval_numerical_gradient


def test_softmax() -> None:
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x)
    loss, dx = softmax_loss(x, y)

    assert np.allclose(loss, 2.3, rtol=1e-2)
    assert np.allclose(dx_num, dx)
