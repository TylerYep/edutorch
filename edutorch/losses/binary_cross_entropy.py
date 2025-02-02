from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from edutorch.nptypes import NPArray, NPIntArray


def binary_cross_entropy(x: NPArray, y: NPIntArray) -> tuple[float, NPArray]:
    """
    Compute binary cross-entropy loss and gradient.
    https://sparrow.dev/binary-cross-entropy/

    Inputs:
    - x: An array with len(y) labels where each is one of {0, 1}
    - y: An array with len(yhat) predictions between [0, 1]

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss = -(y * np.log(x) + (1 - y) * np.log(1 - x)).mean()
    dx = -(y / x) + (1 - y) / (1 - x)
    return loss, dx
