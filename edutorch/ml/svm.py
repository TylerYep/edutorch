from typing import Tuple

import numpy as np

from .linear_classifier import LinearClassifier


class SVMClassifier(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Structured SVM loss function, vectorized implementation.
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
        num_train = X.shape[0]
        score_matrix = X.dot(self.W)
        correct_class_scores = score_matrix[np.arange(num_train), y].reshape(-1, 1)
        margin = score_matrix - correct_class_scores + 1  # note delta = 1
        margin[margin < 0] = 0
        margin[np.arange(num_train), y] = 0
        loss = np.sum(margin)

        margin[margin > 0] = 1
        num_y = np.sum(margin, axis=1)
        margin[np.arange(num_train), y] = -num_y
        dW = X.T.dot(margin)

        return loss, dW

    def svm_loss_naive(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Structured SVM loss function, naive implementation (with loops).
        Inputs and outputs are the same as svm_loss_vectorized.
        """
        dW = np.zeros(self.W.shape)  # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in range(num_train):
            scores = X[i].dot(self.W)
            correct_class_score = scores[y[i]]
            for j in range(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                if margin > 0:
                    loss += margin
                    dW[:, j] += X[i]
                    dW[:, y[i]] -= X[i]

        return loss, dW
