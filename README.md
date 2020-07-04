# edutorch
Rewritten PyTorch framework designed to help you learn AI/ML!

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Build Status](https://travis-ci.com/TylerYep/edutorch.svg?branch=master)](https://travis-ci.com/TylerYep/edutorch)
[![GitHub license](https://img.shields.io/github/license/TylerYep/edutorch)](https://github.com/TylerYep/edutorch/blob/master/LICENSE)

PyTorch is one of the most amazing frameworks for building and training deep neural networks. One of its biggest strengths is providing an intuitive and extendable interface for building and training these models.

In this project, I provide my own version of the PyTorch framework, designed to help you understand the key concepts. The goal is to provide explicit implementations of popular layers, models, and optimizers. Above all else, this code is designed to be readable and clear. Many of these examples are modified from Stanford's CS 230 / 231N course materials available online.

## EduTorch vs PyTorch
One notable difference between EduTorch and PyTorch is that EduTorch _does NOT provide autograd_. There are many educational benefits to deriving/implementing the backprop step yourself, and if you want automatic gradient calculations, you are better off using the real framework. Additionally, if you wanted to learn how the autograd system is implemented, you can check out Andrej Karpathy's [micrograd project](https://github.com/karpathy/micrograd)

Also, there is no CUDA or GPU support.

TODO:
Dataclasses for all modules?