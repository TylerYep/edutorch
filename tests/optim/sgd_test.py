import numpy as np

from edutorch.nn import Linear
from edutorch.optim import SGD


def test_sgd() -> None:
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)

    model = Linear(N, D)
    optimizer = SGD(model, lr=1e-2)

    next_w = optimizer.update(optimizer.context["w"], w, dw)

    expected_next_w = np.asarray(
        [
            [-0.394, -0.34189474, -0.28978947, -0.23768421, -0.18557895],
            [-0.13347368, -0.08136842, -0.02926316, 0.02284211, 0.07494737],
            [0.12705263, 0.17915789, 0.23126316, 0.28336842, 0.33547368],
            [0.38757895, 0.43968421, 0.49178947, 0.54389474, 0.596],
        ]
    )

    assert np.allclose(next_w, expected_next_w)
