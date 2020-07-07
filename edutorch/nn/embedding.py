import numpy as np

from .module import Module


class Embedding(Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.W = np.random.normal(scale=1e-3, size=(input_dim, output_dim))
        self.set_parameters("W")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for word embeddings. We operate on minibatches of size N where
        each sequence has length T. We assume a vocabulary of V words, assigning each
        word to a vector of dimension D.

        Inputs:
        - x: Integer array of shape (N, T) giving indices of words. Each element idx
        of x muxt be in the range 0 <= idx < V.
        - W: Weight matrix of shape (V, D) giving word vectors for all words.

        Returns a tuple of:
        - out: Array of shape (N, T, D) giving word vectors for all input words.
        - cache: Values needed for the backward pass
        """
        self.cache = (x,)
        return self.W[x]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for word embeddings. We cannot back-propagate into the words
        since they are integers, so we only return gradient for the word embedding
        matrix.

        Inputs:
        - dout: Upstream gradients of shape (N, T, D)
        - cache: Values from the forward pass

        Returns:
        - dW: Gradient of word embedding matrix, of shape (V, D).
        """
        (x,) = self.cache
        dW = np.zeros_like(self.W)
        np.add.at(dW, x, dout)
        return dW
