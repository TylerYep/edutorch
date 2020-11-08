# pylint: disable=abstract-method, not-callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LogisticRegression(nn.Module):  # type: ignore
    """ Think of this as Sigmoidal Classification! """

    def __init__(self) -> None:
        super().__init__()
        # define parameters to be part of the model
        # "weight" of linear model
        self.theta1 = nn.Parameter(torch.zeros(1))  # type: ignore
        # "bias" of linear model
        self.theta0 = nn.Parameter(torch.zeros(1))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function is called to apply your function to input. In this case:
            weight * input + bias

        y = sigmoid(theta1 * x + theta0)
        """
        return torch.sigmoid(self.theta1 * x + self.theta0)  # type: ignore


def optimize(
    model: nn.Module,  # type: ignore
    data: torch.Tensor,
    max_iters: int = 100,
    verbose: bool = True,
) -> None:
    # Binds the model to the optimizer.
    # Notice we set a learning rate (lr)! this is really important in
    # machine learning -- try a few different ones and see what happens.
    optimizer = Adam(model.parameters(), lr=0.001)

    # Pytorch expects inputs and outputs of certain shapes
    # (# data, # features). In our case, we only have 1 feature,
    # so the second dimension will be 1. These next two lines
    # transform the data to the right shape!
    x = make_tensor(data, True)
    y = make_tensor(data, False)

    # At the beginning, default minimum loss to infinity
    min_loss = float("inf")
    counter = 0

    while counter < max_iters:
        # Wipe any existing gradients from previous iterations!
        # (don't forget to do this for your own code!)
        optimizer.zero_grad()

        # A "forward" pass through the model. It runs the logistic
        # regression with the current parameters to get a prediction
        pred = model(x)

        # A loss (or objective) function tells us how "good" a
        # prediction is compared to the true answer.
        #
        # This is mathematically equivalent to scoring the truth
        # against a bernoulli distribution with parameters equal
        # to the prediction (a number between 0 and 1).
        loss = F.binary_cross_entropy(pred, y)

        # This step computes all gradients with "autograd"
        # i.e. automatic differentiation
        loss.backward()  # type: ignore

        # This function actually change the parameters
        optimizer.step()

        # if current loss is better than any ones we've seen, save the parameters.
        curr_loss = loss.item()
        counter += 1
        if curr_loss < min_loss:
            best_params = (model.theta1.item(), model.theta0.item())
            min_loss = curr_loss

        if verbose:
            print(
                f"loss = {curr_loss:.4f}, "
                f"c1 = {best_params[0]:.4f}, "
                f"c2 = {best_params[1]:.4f}"
            )


def make_tensor(data: np.array, is_input: bool) -> torch.Tensor:
    """
    Torch is very specific that the input and output have to be lists of matrices.
    Our current input/output is just a single value, so we are going to need
    to make it a n x 1 "tensor".
    """
    dim = 0 if is_input else 1
    return torch.tensor(data[:, dim]).unsqueeze(1).float()  # type: ignore
