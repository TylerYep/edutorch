import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from edutorch.nn import Linear
from edutorch.optim import RMSProp


def test_rmsprop(monkeypatch: MonkeyPatch) -> None:
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    model = Linear(N, D)
    monkeypatch.setattr(RMSProp, "init_context", lambda self, w: (v,))
    optimizer = RMSProp(model, lr=1e-2)

    next_w, (next_v,) = optimizer.update(optimizer.context["w"], w, dw)

    expected_next_w = np.asarray(
        [
            [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
            [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774],
            [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447],
            [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619],
        ]
    )
    expected_velocity = np.asarray(
        [
            [0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321],
            [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377],
            [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936],
            [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926],
        ]
    )

    assert np.allclose(next_w, expected_next_w)
    assert np.allclose(next_v, expected_velocity)
