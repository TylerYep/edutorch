import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from edutorch.nn import Linear
from edutorch.optim import Adam


def test_adam(monkeypatch: MonkeyPatch) -> None:
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
    v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)

    model = Linear(N, D)
    monkeypatch.setattr(Adam, "init_context", lambda self, w: (m, v, 5))
    optimizer = Adam(model, lr=1e-2)

    next_w, (next_m, next_v, _) = optimizer.update(optimizer.context["w"], w, dw)

    expected_next_w = np.asarray(
        [
            [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
            [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
            [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
            [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459],
        ]
    )
    expected_v = np.asarray(
        [
            [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853],
            [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385],
            [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767],
            [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966],
        ]
    )
    expected_m = np.asarray(
        [
            [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
            [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
            [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
            [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85],
        ]
    )

    assert np.allclose(next_w, expected_next_w)
    assert np.allclose(next_m, expected_m)
    assert np.allclose(next_v, expected_v)
