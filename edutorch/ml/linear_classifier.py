from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from edutorch.nptypes import NPArray, NPIntArray


class LinearClassifier(ABC):
    """A simple linear classifier abstract class."""

    def __init__(
        self, X: NPArray, y: NPIntArray, learning_rate: float = 1e-3, reg: float = 1e-5
    ) -> None:
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_iters = 100
        batch_size = 200
        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = int(np.max(y) + 1)
        self.W = 0.001 * np.random.randn(dim, num_classes)
        self.reg = reg

        # Run stochastic gradient descent to optimize W
        for it in range(num_iters):
            # Sample batch_size elements from the training data and their
            # corresponding labels to use in this round of gradient descent.
            # Store the data in X_batch and their corresponding labels in
            # y_batch; after sampling X_batch should have shape (batch_size, dim)
            # and y_batch should have shape (batch_size,)
            inds = np.random.choice(num_train, batch_size)
            X_batch = X[inds]
            y_batch = y[inds]

            # Evaluate loss and gradient
            loss, grad = self.regularize(*self.loss(X_batch, y_batch), num_train)
            self.W -= learning_rate * grad

            if it % 100 == 0:
                print(f"iteration {it} / {num_iters}: loss {loss}")

    def predict(self, X: NPArray) -> NPArray:
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a
            1-dimensional array of length N, and each element is
            an integer giving the predicted class.
        """
        return np.argmax(np.dot(X, self.W), axis=1)

    @abstractmethod
    def loss(self, X: NPArray, y: NPIntArray) -> tuple[float, NPArray]:
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X: A numpy array of shape (N, D) containing a minibatch of N
            data points; each point has dimension D.
        - y: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        del X, y
        return 0, np.zeros_like(self.W)

    def regularize(
        self, loss: float, dW: NPArray, num_train: int
    ) -> tuple[float, NPArray]:
        """
        Right now, the loss is a sum over all training examples, but we want it
        to be an average instead so we divide by num_train.
        We then add regularization to the loss.
        """
        loss /= num_train
        dW /= num_train

        loss += self.reg * np.sum(self.W**2)
        dW += self.reg * 2 * self.W
        return loss, dW
