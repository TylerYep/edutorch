from typing import Tuple

import numpy as np
import pytest
from torchvision import datasets


@pytest.fixture(scope="session")
def fashion_mnist(num_train: int = 100, num_test: int = 10) -> Tuple[np.ndarray, ...]:
    train_set = datasets.FashionMNIST("data/", train=True, download=True)
    test_set = datasets.FashionMNIST("data/", train=False, download=True)

    X_train, y_train = train_set.data.numpy().astype(float), train_set.targets.numpy()
    X_test, y_test = test_set.data.numpy().astype(float), test_set.targets.numpy()

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
