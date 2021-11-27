import numpy as np
import pytest

from edutorch.ml.logistic_regression import LogisticRegression, optimize


@pytest.mark.skip(reason="We don't use torch anymore")
def test_logistic_regression() -> None:
    data = np.genfromtxt("data/logRegData.csv", delimiter=",")
    model = LogisticRegression()
    optimize(model, data, verbose=False)
