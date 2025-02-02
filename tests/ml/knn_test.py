from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from edutorch.ml.knn import KNearestNeighbors

if TYPE_CHECKING:
    from edutorch.nptypes import NPArray, NPIntArray


@pytest.mark.skip(reason="Takes too long to run")
def test_knn_compute_distance(
    fashion_mnist: tuple[NPArray, NPIntArray, NPArray, NPIntArray],
) -> None:
    X_train, y_train, X_test, _ = fashion_mnist

    knn = KNearestNeighbors(X_train, y_train)

    distsA = knn.compute_distances(X_test, 0)
    distsB = knn.compute_distances(X_test, 1)
    assert np.linalg.norm(distsA - distsB) < 0.001


# def test_knn_predict_labels(fashion_mnist) -> None:
#     pass
