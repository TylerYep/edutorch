from __future__ import annotations

import random

import numpy as np
import pytest
from mnist import fashion_mnist as FashionMNIST  # type: ignore[import]  # noqa: N812

from edutorch.typing import NPArray, NPIntArray


@pytest.fixture(autouse=True)
def _set_random_seed() -> None:
    random.seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def fashion_mnist(
    num_train: int = 100, num_test: int = 10
) -> tuple[NPArray, NPIntArray, NPArray, NPIntArray]:
    X_train, y_train, X_test, y_test = FashionMNIST(cache="data/FashionMNIST")

    mask = list(range(num_train))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    X_train = np.reshape(X_train, (num_train, -1))
    X_test = np.reshape(X_test, (num_test, -1))

    return X_train, y_train, X_test, y_test
