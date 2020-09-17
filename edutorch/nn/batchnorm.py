from typing import Tuple

import numpy as np

from .module import Module


class BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        train_mode: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.train_mode = train_mode
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.set_parameters("running_mean", "running_var", "gamma", "beta")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization.
        Uses minibatch statistics to compute the mean and variance, use
        these statistics to normalize the incoming data, and scale and
        shift the normalized data using gamma and beta.

        During training the sample mean and (uncorrected) sample variance are
        computed from minibatch statistics and used to normalize the incoming data.
        During training we also keep an exponentially decaying running mean of the mean
        and variance of each feature, and these averages are used to normalize data
        at test-time.

        At each timestep we update the running averages for mean and variance using
        an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that though you should be keeping track of the running
        variance, you should normalize the data based on the standard
        deviation (square root of variance) instead!
        Referencing the original paper (https://arxiv.org/abs/1502.03167)
        might prove to be helpful.

        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using a
        large number of training images rather than using a running average. For
        this implementation we have chosen to use running averages instead since
        they do not require an additional estimation step; the torch7 implementation
        of batch normalization also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - momentum: Constant for running mean / variance.
            - running_mean: Array of shape (D,) giving running mean of features
            - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        if self.train_mode:
            # Compute output
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + self.eps)
            xn = xc / std
            out = self.gamma * xn + self.beta

            # Update running average of mean
            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * mu

            # Update running average of variance
            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * var

        else:
            # Using running mean and variance to normalize
            std = np.sqrt(self.running_var + self.eps)
            xn = (x - self.running_mean) / std
            out = self.gamma * xn + self.beta

        self.cache = (xn, std)

        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Alternative backward pass for batch normalization.

        For this implementation you should work out the derivatives for the batch
        normalizaton backward pass on paper and simplify as much as possible. You
        should be able to derive a simple expression for the backward pass.
        See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same cache variable
        as batchnorm_backward, but might not use all of the values in the cache.

        If you get stuck, check out a worked-out solution:
        https://kevinzakka.github.io/2016/09/14/batch_normalization/

        Inputs / outputs: Same as batchnorm_backward
        """
        xn, std = self.cache

        dx_hat = dout * self.gamma
        dx = (
            dx_hat - np.mean(dx_hat, axis=0) - xn * np.mean(dx_hat * xn, axis=0)
        ) / std

        dgamma = np.sum(dout * xn, axis=0)
        dbeta = np.sum(dout, axis=0)

        return dx, dgamma, dbeta

    def backward_naive(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a computation graph for
        batch normalization on paper and propagate gradients backward through
        intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
        """
        xn, std = self.cache

        if self.train_mode:
            N = dout.shape[0]
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(xn * dout, axis=0)
            dxn = self.gamma * dout
            dxc = dxn / std
            dstd = -np.sum((dxn * xn) / std, axis=0)
            dvar = 0.5 * dstd / std
            dxc += (2.0 / N) * (xn * std) * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / N

        else:
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(xn * dout, axis=0)
            dxn = self.gamma * dout
            dx = dxn / std

        return dx, dgamma, dbeta
