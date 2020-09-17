import numpy as np

from edutorch.nn import BatchNorm
from tests.gradient_check import estimate_gradients


def test_batchnorm_forward() -> None:
    D = 3
    a = np.random.randn(200, D)
    model = BatchNorm(D)

    a_norm = model(a)

    assert np.allclose(
        a_norm.mean(axis=0), np.zeros(3)
    ), "After batch norm (gamma=1, beta=0), means should be close to 0."
    assert np.allclose(
        a_norm.std(axis=0), np.ones(3)
    ), "After batch norm (gamma=1, beta=0), stds should be close to 1."


def test_batchnorm_forward_preset_gamma_beta() -> None:
    D = 3
    a = np.random.randn(200, D)
    model = BatchNorm(D)
    model.gamma = np.asarray([1.0, 2.0, 3.0])
    model.beta = np.asarray([11.0, 12.0, 13.0])

    a_norm = model(a)

    assert np.allclose(a_norm.mean(axis=0), model.beta), (
        f"After batch norm (gamma={model.gamma}, "
        f"beta={model.beta}), means should be close to beta."
    )
    assert np.allclose(a_norm.std(axis=0), model.gamma), (
        f"After batch norm (gamma={model.gamma}, "
        f"beta={model.beta}), std should be close to gamma."
    )


def test_batchnorm_backward() -> None:
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    model = BatchNorm(D)

    params = {"gamma": gamma, "beta": beta}
    dx_num, dgamma_num, dbeta_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dgamma, dbeta = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dgamma_num, dgamma)
    assert np.allclose(dbeta_num, dbeta)


def test_batchnorm_backward_naive() -> None:
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    model = BatchNorm(D)

    params = {"gamma": gamma, "beta": beta}
    dx_num, dgamma_num, dbeta_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dgamma, dbeta = model.backward_naive(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dgamma_num, dgamma)
    assert np.allclose(dbeta_num, dbeta)
