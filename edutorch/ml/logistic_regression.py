from typing import Any, override

import numpy as np

from edutorch.losses import binary_cross_entropy
from edutorch.nn import Module
from edutorch.nn.functional import sigmoid
from edutorch.optim import Adam
from edutorch.typing import NPArray


class LogisticRegression(Module):
    """Think of this as Sigmoidal Classification!"""

    def __init__(self) -> None:
        super().__init__()
        # define parameters to be part of the model
        self.theta = np.zeros(2)  # ("bias", "weight") of linear model

    @override
    def forward(self, x: NPArray) -> NPArray:
        """
        This function is called to apply your function to input. In this case:
            weight * input + bias

        y = sigmoid(theta1 * x + theta0)
        """
        theta0, theta1 = self.theta[0], self.theta[1]
        return sigmoid(theta1 * x + theta0)

    @override
    def backward(self, dout: NPArray) -> Any:
        return dout


def optimize(
    model: LogisticRegression, data: NPArray, max_iters: int = 100, verbose: bool = True
) -> None:
    # Binds the model to the optimizer.
    # Notice we set a learning rate (lr)! this is really important in
    # machine learning -- try a few different ones and see what happens.
    optimizer = Adam(model, lr=0.001)

    # Pytorch expects inputs and outputs of certain shapes
    # (# data, # features). In our case, we only have 1 feature,
    # so the second dimension will be 1. These next two lines
    # transform the data to the right shape!
    x = np.array(data[:, 0])
    y = np.array(data[:, 1]).dtype("int64")  # pylint: disable=not-callable

    # At the beginning, default minimum loss to infinity
    min_loss = float("inf")
    counter = 0
    best_params = (model.theta[0], model.theta[1])
    while counter < max_iters:
        # Wipe any existing gradients from previous iterations!
        # (don't forget to do this for your own code!)
        # optimizer.zero_grad()

        # A "forward" pass through the model. It runs the logistic
        # regression with the current parameters to get a prediction
        pred = model(x)

        # A loss (or objective) function tells us how "good" a
        # prediction is compared to the true answer.
        #
        # This is mathematically equivalent to scoring the truth
        # against a bernoulli distribution with parameters equal
        # to the prediction (a number between 0 and 1).
        loss, dx = binary_cross_entropy(pred, y)
        # loss = F.binary_cross_entropy(pred, y)

        # This step computes all gradients with "autograd"
        # i.e. automatic differentiation
        grads = model.backward(dx)

        # This function actually change the parameters
        optimizer.step(model, grads)

        # if current loss is better than any ones we've seen, save the parameters.
        counter += 1
        if loss < min_loss:
            best_params = (model.theta[0], model.theta[1])
            min_loss = loss

        if verbose:
            print(
                f"loss = {loss:.4f}, "
                f"c1 = {best_params[0]:.4f}, "
                f"c2 = {best_params[1]:.4f}"
            )
