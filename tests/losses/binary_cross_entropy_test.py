import numpy as np

from edutorch.losses import binary_cross_entropy

# from tests.gradient_check import eval_numerical_gradient


def test_binary_cross_entropy() -> None:
    loss, dx = binary_cross_entropy(np.array([0.5]), np.array([0]))

    assert np.allclose(loss, np.array([0.6931471805599453]))
    assert np.allclose(dx, np.array([2]))

    # num_inputs = 50
    # x = np.random.uniform(0, 1, num_inputs)
    # y = np.random.randint(2, size=num_inputs)  # 0 or 1
    # print(x, y)

    # dx_num = eval_numerical_gradient(lambda x: binary_cross_entropy(x, y)[0], x)
    # loss, dx = binary_cross_entropy(x, y)

    # print(dx_num, dx)
    # assert np.allclose(loss, _, rtol=1e-2), loss
    # assert np.allclose(dx_num, dx)
