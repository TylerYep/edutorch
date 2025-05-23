from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from edutorch.ml.linear_classifier import LinearClassifier

if TYPE_CHECKING:
    from edutorch.nptypes import NPArray, NPIntArray


def softmax(f: NPArray) -> NPArray:
    """Numerically stable implementation of softmax."""
    f -= np.max(f)
    exp_f = np.exp(f)
    if len(f.shape) == 1:
        return exp_f / np.sum(exp_f, axis=0)
    return exp_f / np.sum(exp_f, axis=1).reshape(-1, 1)


class SoftmaxClassifier(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function."""

    @override
    def loss(self, X: NPArray, y: NPIntArray) -> tuple[float, NPArray]:
        """
        Softmax loss function, vectorized version.
        Inputs and outputs are the same as softmax_loss_naive.
        """
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        scores = X.dot(self.W)
        softmx = softmax(scores)

        loss = np.sum(np.log(softmx[np.arange(num_train), y]))

        kronecker = np.zeros((num_train, num_classes))
        kronecker[np.arange(num_train), y] = 1
        dW = X.T.dot(kronecker - softmx)

        return loss, dW

    def softmax_loss_naive(self, X: NPArray, y: NPIntArray) -> tuple[float, NPArray]:
        """
        Softmax loss function, naive implementation (with loops)
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        for i in range(num_train):
            scores = X[i].dot(self.W)
            softmx = softmax(scores)
            loss -= np.log(softmx[y[i]])

            for j in range(num_classes):
                if j == y[i]:
                    dW[:, j] -= (1 - softmx[j]) * X[i]
                else:
                    dW[:, j] += softmx[j] * X[i]

        return loss, dW
