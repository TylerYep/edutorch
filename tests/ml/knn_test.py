from __future__ import annotations

import numpy as np
import pytest

from edutorch.ml.knn import KNearestNeighbors
from edutorch.typing import NPArray


@pytest.mark.skip(reason="Takes too long to run")
def test_knn_compute_distance(fashion_mnist: tuple[NPArray, ...]) -> None:
    X_train, y_train, X_test, _ = fashion_mnist

    knn = KNearestNeighbors(X_train, y_train)

    distsA = knn.compute_distances(X_test, 0)
    distsB = knn.compute_distances(X_test, 1)
    assert np.linalg.norm(distsA - distsB) < 0.001


# def test_knn_predict_labels(fashion_mnist) -> None:
#     pass
