import numpy as np

from edutorch.nn import LayerNorm
from tests.gradient_check import estimate_gradients


def test_layernorm_forward() -> None:
    N, D1, D2, D3 = 4, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    model = LayerNorm(D3)

    a_norm = model(a)

    assert np.allclose(
        a_norm.mean(axis=1), np.zeros(4)
    ), "After layer norm (gamma=1, beta=0), means should be close to 0."
    assert np.allclose(
        a_norm.std(axis=1), np.ones(4)
    ), "After layer norm (gamma=1, beta=0), stds should be close to 1."


def test_layernorm_forward_preset_gamma_beta() -> None:
    N, D1, D2, D3 = 4, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    model = LayerNorm(D3)
    model.gamma = np.asarray([3.0] * 3)
    model.beta = np.asarray([5.0] * 3)
    gamma = np.asarray([3.0] * 4)
    beta = np.asarray([5.0] * 4)

    a_norm = model(a)

    assert np.allclose(a_norm.mean(axis=1), beta), (
        f"After layer norm (gamma={model.gamma}, "
        f"beta={model.beta}), means should be close to beta."
    )
    assert np.allclose(a_norm.std(axis=1), gamma), (
        f"After layer norm (gamma={model.gamma}, "
        f"beta={model.beta}), std should be close to gamma."
    )


def test_layernorm_backward() -> None:
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    model = LayerNorm(D)

    params = {"gamma": gamma, "beta": beta}
    dx_num, dgamma_num, dbeta_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dgamma, dbeta = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dgamma_num, dgamma)
    assert np.allclose(dbeta_num, dbeta)
