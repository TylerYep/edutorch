import numpy as np

from edutorch.ml.logistic_regression import LogisticRegression, optimize


def test_logistic_regression() -> None:
    data = np.genfromtxt("data/logRegData.csv", delimiter=",")
    model = LogisticRegression()
    optimize(model, data, verbose=False)
