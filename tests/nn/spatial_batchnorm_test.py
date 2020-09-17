import numpy as np

from edutorch.nn import SpatialBatchNorm
from tests.gradient_check import estimate_gradients


def test_spatial_batchnorm_forward() -> None:
    N, C, H, W = 2, 3, 4, 5
    x = 4 * np.random.randn(N, C, H, W) + 10
    model = SpatialBatchNorm(C)

    x_norm = model(x)

    assert np.allclose(
        x_norm.mean(axis=(0, 2, 3)), np.zeros(3)
    ), "After batch norm (gamma=1, beta=0), means should be close to 0."
    assert np.allclose(
        x_norm.std(axis=(0, 2, 3)), np.ones(3)
    ), "After batch norm (gamma=1, beta=0), stds should be close to 1."


def test_spatial_batchnorm_forward_preset_gamma_beta() -> None:
    N, C, H, W = 2, 3, 4, 5
    x = 4 * np.random.randn(N, C, H, W) + 10
    model = SpatialBatchNorm(C)
    model.gamma = np.asarray([3, 4, 5])
    model.beta = np.asarray([6, 7, 8])

    x_norm = model(x)

    assert np.allclose(x_norm.mean(axis=(0, 2, 3)), model.beta), (
        f"After spatial batch norm (gamma={model.gamma}, "
        f"beta={model.beta}), means should be ~beta."
    )
    assert np.allclose(x_norm.std(axis=(0, 2, 3)), model.gamma), (
        f"After spatial batch norm (gamma={model.gamma}, "
        f"beta={model.beta}), std should be ~gamma."
    )


def test_spatial_batchnorm_backward() -> None:
    N, C, H, W = 2, 3, 4, 5
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(C)
    beta = np.random.randn(C)
    dout = np.random.randn(N, C, H, W)

    model = SpatialBatchNorm(C)

    params = {"gamma": gamma, "beta": beta}
    dx_num, dgamma_num, dbeta_num = estimate_gradients(model, dout, x, params)

    _ = model(x)
    dx, dgamma, dbeta = model.backward(dout)

    assert np.allclose(dx_num, dx)
    assert np.allclose(dgamma_num, dgamma)
    assert np.allclose(dbeta_num, dbeta)
