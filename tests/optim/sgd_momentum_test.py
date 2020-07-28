import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from edutorch.nn import Linear
from edutorch.optim import SGDMomentum


def test_sgd_momentum(monkeypatch: MonkeyPatch) -> None:
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    model = Linear(N, D)
    monkeypatch.setattr(SGDMomentum, "init_context", lambda self, w: (0.9, v))
    optimizer = SGDMomentum(model, lr=1e-3)

    next_w, (_, next_v) = optimizer.update(optimizer.context["w"], w, dw)

    expected_next_w = np.asarray(
        [
            [0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
            [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
            [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
            [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096],
        ]
    )
    expected_velocity = np.asarray(
        [
            [0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
            [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
            [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
            [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096],
        ]
    )

    assert np.allclose(next_w, expected_next_w)
    assert np.allclose(next_v, expected_velocity)
