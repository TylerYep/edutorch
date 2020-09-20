import random
from typing import Callable, Dict, List

import numpy as np

from edutorch.nn import Module


def rel_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def estimate_gradients(
    model: Module, dout: np.ndarray, x: np.ndarray, kwparams: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """
    Gets the gradient estimate for all parameters of the model. Overrides each
    parameter of the model using the values in kwparams.
    """

    def grad_fn(
        model: Module, x: np.ndarray, **kwargs: Dict[str, np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a grad function that takes in an input z and sets all attributes
        of the model to the kwargs, except for the target value z.
        """

        def dx_or_other_fn(z: np.ndarray) -> np.ndarray:
            """ Function used as input to eval_numerical_gradient. """
            for key, val in kwargs.items():
                setattr(model, key, z if val is None else val)
            return model(z if x is None else x)

        return dx_or_other_fn

    # Add x as a special case
    fx = grad_fn(model, None, **kwparams)
    approx_derivatives = [eval_numerical_gradient_array(fx, x, dout)]
    for name, param in kwparams.items():
        # Shallow copy works here because we replace the target element with None.
        new_kwargs = dict(kwparams)
        new_kwargs[name] = None

        # the numerical gradient function with respect to variable _name_
        f_name = grad_fn(model, x, **new_kwargs)
        approx_derivatives.append(
            eval_numerical_gradient_array(f_name, param.copy(), dout)
        )
    return approx_derivatives


def eval_numerical_gradient(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    verbose: bool = False,
    h: float = 1e-5,
) -> np.ndarray:
    """
    A naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    # fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def eval_numerical_gradient_array(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    df: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


# The following functions have yet to be used.


def eval_numerical_gradient_blobs(
    f: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    output: np.ndarray,
    h: float = 1e-5,
) -> List[np.ndarray]:
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a
    blob where outputs will be written. For example, f might be called like:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def grad_check_sparse(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    analytic_grad: np.ndarray,
    num_checks: int = 10,
    h: float = 1e-5,
) -> None:
    """
    Sample a few random elements and only return
    numerical gradients in those dimensions.
    """
    for _ in range(num_checks):
        ix = tuple(random.randrange(m) for m in x.shape)

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            f"numerical: {grad_numerical}, "
            f"analytic: {grad_analytic}, "
            f"relative error: {error}"
        )
