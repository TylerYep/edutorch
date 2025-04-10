import numpy as np

from edutorch.nptypes import NPArray, NPIntArray


class KNearestNeighbors:
    """A k-NN classifier with L2 distance"""

    def __init__(self, X_train: NPArray, y_train: NPIntArray, k: int = 3) -> None:
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Remember that training a kNN classifier is a noop:
        the Classifier simply remembers the data and does no further processing.
        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
            consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
            y[i] is the label for X[i].
        - k: The number of nearest neighbors that vote for the predicted labels.
        """
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X: NPArray) -> NPArray:
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
            of num_test samples each of dimension D.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].
        """
        dists = self.compute_distances(X)
        return self.predict_labels(dists)

    def compute_distances(self, X: NPArray, num_loops: int = 0) -> NPArray:
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth training
            point.
        """
        if num_loops == 0:
            return np.sqrt(
                np.sum(X**2, axis=1).reshape(-1, 1)
                - 2 * np.dot(X, self.X_train.T)
                + np.sum(self.X_train**2, axis=1)
            )

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        if num_loops == 1:
            for i in range(num_test):
                dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))

        elif num_loops == 2:
            for i in range(num_test):
                for j in range(num_train):
                    dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

        return dists

    def predict_labels(self, dists: NPArray) -> NPArray:
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # List storing the labels of the k nearest neighbors to the ith test point.
            inds = np.argsort(dists[i, :])[: self.k]
            closest_y = self.y_train[inds]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred
