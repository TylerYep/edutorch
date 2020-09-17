from typing import Tuple

import numpy as np

from .module import Module


class MaxPool2d(Module):
    def __init__(
        self, kernel_size: Tuple[int, int], stride: int = 1, pad: int = 2
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions

        No padding is necessary here. Output size is given by

        Returns a tuple of:
        - out: Output data, of shape (N, C, H', W') where H' and W' are given by
        H' = 1 + (H - pool_height) / stride
        W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        pool_height, pool_width = self.kernel_size
        stride = self.stride

        N, C, H, W = x.shape
        H_prime = 1 + (H - pool_height) // stride
        W_prime = 1 + (W - pool_width) // stride

        out = np.zeros((N, C, H_prime, W_prime))
        for n in range(N):
            for c in range(C):
                for i in range(0, H_prime * stride, stride):
                    for j in range(0, W_prime * stride, stride):
                        out[n, c, i // stride, j // stride] = np.amax(
                            x[n, c, i : i + pool_height, j : j + pool_width]
                        )
        self.cache = (x,)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        A naive implementation of the backward pass for a max-pooling layer.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """
        (x,) = self.cache
        pool_height, pool_width = self.kernel_size
        stride = self.stride

        N, C, H, W = x.shape
        H_prime = 1 + (H - pool_height) // stride
        W_prime = 1 + (W - pool_width) // stride

        dx = np.zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for i in range(0, H_prime * stride, stride):
                    for j in range(0, W_prime * stride, stride):
                        a = x[n, c, i : i + pool_height, j : j + pool_width]
                        k = np.unravel_index(a.argmax(), a.shape)
                        dx[n, c, i : i + pool_height, j : j + pool_width][k] = dout[
                            n, c, i // stride, j // stride
                        ]
        return dx
