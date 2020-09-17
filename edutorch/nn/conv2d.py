from typing import Tuple

import numpy as np

from .module import Module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        pad: int = 2,
    ) -> None:
        super().__init__()
        self.w = np.random.normal(
            scale=1e-3, size=(out_channels, in_channels, *kernel_size)
        )
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.pad = pad
        self.set_parameters("w", "b")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.


        During padding, 'pad' zeros should be placed symmetrically
        (i.e equally on both sides) along the height and width axes of the input.
        Be careful not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        N, _, H, W = x.shape
        F, _, HH, WW = self.w.shape
        pad = self.pad
        stride = self.stride

        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        out = np.zeros((N, F, H_prime, W_prime))

        for n in range(N):
            for f in range(F):
                for i in range(0, H_prime * stride, stride):
                    for j in range(0, W_prime * stride, stride):
                        out[n, f, i // stride, j // stride] = (
                            np.sum(self.w[f] * x_padded[n, :, i : i + HH, j : j + WW])
                            + self.b[f]
                        )
        self.cache = (x,)
        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        (x,) = self.cache
        dx = np.zeros_like(x)
        dw = np.zeros_like(self.w)
        db = np.zeros_like(self.b)

        N, _, _, _ = x.shape
        F, _, HH, WW = self.w.shape
        pad = self.pad
        stride = self.stride

        _, _, H_prime, W_prime = dout.shape
        padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
        x_padded = np.pad(x, padding, mode="constant")
        dx_padded = np.pad(dx, padding, mode="constant")

        for n in range(N):
            for f in range(F):
                for i in range(0, H_prime * stride, stride):
                    for j in range(0, W_prime * stride, stride):
                        do = dout[n, f, i // stride, j // stride]
                        dx_padded[n, :, i : i + HH, j : j + WW] += do * self.w[f]
                        dw[f] += do * x_padded[n, :, i : i + HH, j : j + WW]
                        db[f] += do
        dx = dx_padded[:, :, pad:-pad, pad:-pad]

        return dx, dw, db
