import numpy as np

from edutorch.layers import BatchNorm
from gradient_check import eval_numerical_gradient_array, rel_error


def test_batchnorm_forward():
    a = np.random.randn(200, 3)
    model = BatchNorm(3)

    a_norm = model(a)

    assert np.allclose(
        a_norm.mean(axis=0), np.zeros(3)
    ), "After batch norm (gamma=1, beta=0), means should be close to 0."
    assert np.allclose(
        a_norm.std(axis=0), np.ones(3)
    ), "After batch norm (gamma=1, beta=0), stds should be close to 1."


def test_batchnorm_forward_preset_gamma_beta():
    a = np.random.randn(200, 3)
    model = BatchNorm(3)
    model.gamma = np.asarray([1.0, 2.0, 3.0])
    model.beta = np.asarray([11.0, 12.0, 13.0])

    a_norm = model(a)

    assert np.allclose(
        a_norm.mean(axis=0), model.beta
    ), f"After batch norm (gamma={model.gamma}, beta={model.beta}), means should be close to beta."
    assert np.allclose(
        a_norm.std(axis=0), model.gamma
    ), f"After batch norm (gamma={model.gamma}, beta={model.beta}), std should be close to gamma."


# def test_batchnorm_backward():
#     # Gradient check batchnorm backward pass
#     np.random.seed(231)
#     N, D = 4, 5
#     x = 5 * np.random.randn(N, D) + 12
#     gamma = np.random.randn(D)
#     beta = np.random.randn(D)
#     dout = np.random.randn(N, D)

#     model = BatchNorm(D)
#     fx = lambda x: model(x, gamma, beta)
#     fg = lambda a: model(x, a, beta)
#     fb = lambda b: model(x, gamma, b)

#     dx_num = eval_numerical_gradient_array(fx, x, dout)
#     da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
#     db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

#     _ = model(x, gamma, beta)
#     dx, dgamma, dbeta = model.backward(dout)
#     # You should expect to see relative errors between 1e-13 and 1e-8
#     print('dx error: ', rel_error(dx_num, dx))
#     print('dgamma error: ', rel_error(da_num, dgamma))
#     print('dbeta error: ', rel_error(db_num, dbeta))
