from typing import Callable, Dict, List

import numpy as np

from edutorch.nn import LSTMCell, Module
from tests.gradient_check import eval_numerical_gradient_array, rel_error


def test_lstm_cell_forward() -> None:
    N, D, H = 3, 4, 5
    x = np.linspace(-0.4, 1.2, num=N * D).reshape(N, D)

    model = LSTMCell(
        prev_h=np.linspace(-0.3, 0.7, num=N * H).reshape(N, H),
        prev_c=np.linspace(-0.4, 0.9, num=N * H).reshape(N, H),
        Wx=np.linspace(-2.1, 1.3, num=4 * D * H).reshape(D, 4 * H),
        Wh=np.linspace(-0.7, 2.2, num=4 * H * H).reshape(H, 4 * H),
        b=np.linspace(0.3, 0.7, num=4 * H),
    )

    next_h, next_c = model(x)
    expected_next_h = np.asarray(
        [
            [0.24635157, 0.28610883, 0.32240467, 0.35525807, 0.38474904],
            [0.49223563, 0.55611431, 0.61507696, 0.66844003, 0.7159181],
            [0.56735664, 0.66310127, 0.74419266, 0.80889665, 0.858299],
        ]
    )
    expected_next_c = np.asarray(
        [
            [0.32986176, 0.39145139, 0.451556, 0.51014116, 0.56717407],
            [0.66382255, 0.76674007, 0.87195994, 0.97902709, 1.08751345],
            [0.74192008, 0.90592151, 1.07717006, 1.25120233, 1.42395676],
        ]
    )

    assert np.allclose(expected_next_h, next_h)
    assert np.allclose(expected_next_c, next_c)


def estimate_gradients(
    model: Module,
    dnext_h: np.ndarray,
    dnext_c: np.ndarray,
    x: np.ndarray,
    kwparams: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """
    Gets the gradient estimate for all parameters of the model. Overrides each
    parameter of the model using the values in kwparams.
    """

    def grad_fn(
        model: Module, x: np.ndarray, h_or_c: str, **kwargs: Dict[str, np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a grad function that takes in an input z and sets all attributes
        of the model to the kwargs, except for the target value z.
        """

        def dx_or_other_fn(z: np.ndarray) -> np.ndarray:
            """ Function used as input to eval_numerical_gradient. """
            for key, val in kwargs.items():
                setattr(model, key, z if val is None else val)
            return model(z if x is None else x)[0 if h_or_c == "h" else 1]

        return dx_or_other_fn

    # Add x as a special case
    approx_derivatives = [
        eval_numerical_gradient_array(grad_fn(model, None, "h", **kwparams), x, dnext_h)
        + eval_numerical_gradient_array(
            grad_fn(model, None, "c", **kwparams), x, dnext_c
        )
    ]
    for name, param in kwparams.items():
        # Shallow copy works here because we replace the target element with None.
        new_kwargs = dict(kwparams)
        new_kwargs[name] = None
        approx_derivatives.append(
            eval_numerical_gradient_array(
                grad_fn(model, x, "h", **new_kwargs), param.copy(), dnext_h
            )
            + eval_numerical_gradient_array(
                grad_fn(model, x, "c", **new_kwargs), param.copy(), dnext_c
            )
        )
    return approx_derivatives


def test_lstm_cell_backward() -> None:
    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    prev_h = np.random.randn(N, H)
    prev_c = np.random.randn(N, H)
    Wx = np.random.randn(D, 4 * H)
    Wh = np.random.randn(H, 4 * H)
    b = np.random.randn(4 * H)

    dnext_h = np.random.randn(*prev_h.shape)
    dnext_c = np.random.randn(*prev_c.shape)

    model = LSTMCell(prev_h=prev_h, prev_c=prev_c, Wx=Wx, Wh=Wh, b=b)

    params = {"prev_h": prev_h, "prev_c": prev_c, "Wx": Wx, "Wh": Wh, "b": b}
    dx_num, dh_num, dc_num, dWx_num, dWh_num, db_num = estimate_gradients(
        model, dnext_h, dnext_c, x, params
    )

    _ = model(x)

    dx, dh, dc, dWx, dWh, db = model.backward((dnext_h, dnext_c))

    assert np.allclose(dx_num, dx), f"dx error: {rel_error(dx_num, dx)}"
    assert np.allclose(dh_num, dh), f"dh error: {rel_error(dh_num, dh)}"
    assert np.allclose(dc_num, dc), f"dc error: {rel_error(dc_num, dc)}"
    assert np.allclose(dWx_num, dWx), f"dWx error: {rel_error(dWx_num, dWx)}"
    assert np.allclose(dWh_num, dWh), f"dWh error: {rel_error(dWh_num, dWh)}"
    assert np.allclose(db_num, db), f"db error: {rel_error(db_num, db)}"
