import numpy as np

from edutorch.nn import SpatialGroupNorm
from tests.gradient_check import estimate_gradients


def test_spatial_groupnorm_forward() -> None:
    N, C, H, W, G = 2, 6, 4, 5, 2
    x = 4 * np.random.randn(N, C, H, W) + 10
    model = SpatialGroupNorm(C, G)
    model.gamma = np.ones((1, C, 1, 1))
    model.beta = np.zeros((1, C, 1, 1))

    out = model(x)
    out_g = out.reshape((N * G, -1))

    assert np.allclose(
        out_g.mean(axis=1), np.zeros(4)
    ), "After batch norm (gamma=1, beta=0), means should be close to 0."
    assert np.allclose(
        out_g.std(axis=1), np.ones(4)
    ), "After batch norm (gamma=1, beta=0), stds should be close to 1."


def test_spatial_groupnorm_backward() -> None:
    N, C, H, W, G = 2, 6, 4, 5, 2
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(1, C, 1, 1)
    beta = np.random.randn(1, C, 1, 1)
    dout = np.random.randn(N, C, H, W)

    model = SpatialGroupNorm(C, G)

    params = {"gamma": gamma, "beta": beta}
    dx_num, dgamma_num, dbeta_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dgamma, dbeta = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dgamma_num, dgamma)
    assert np.allclose(dbeta_num, dbeta)
