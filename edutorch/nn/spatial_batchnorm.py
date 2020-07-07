from typing import Tuple

import numpy as np

from .batchnorm import BatchNorm


class SpatialBatchNorm(BatchNorm):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - momentum: Constant for running mean / variance. momentum=0 means that
                old information is discarded completely at every time step, while
                momentum=1 means that new information is never incorporated. The
                default of momentum=0.9 should work well in most situations.
            - running_mean: Array of shape (D,) giving running mean of features
            - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        N, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        out_flat = super().forward(x_flat)
        out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Computes the backward pass for spatial batch normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        N, C, H, W = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        dx_flat, dgamma, dbeta = super().backward(dout_flat)
        dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return dx, dgamma, dbeta
