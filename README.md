# edutorch

Rewritten PyTorch framework designed to help you learn AI/ML!

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyPI version](https://badge.fury.io/py/edutorch.svg)](https://badge.fury.io/py/edutorch)
[![Build Status](https://github.com/TylerYep/edutorch/actions/workflows/test.yml/badge.svg)](https://github.com/TylerYep/edutorch/actions/workflows/test.yml)
[![GitHub license](https://img.shields.io/github/license/TylerYep/edutorch)](https://github.com/TylerYep/edutorch/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/TylerYep/edutorch/branch/main/graph/badge.svg)](https://codecov.io/gh/TylerYep/edutorch)
[![Downloads](https://pepy.tech/badge/edutorch)](https://pepy.tech/project/edutorch)

PyTorch is one of the most amazing frameworks for building and training deep neural networks. One of its biggest strengths is providing an intuitive and extendable interface for building and training these models.

In this project, I provide my own version of the PyTorch framework, designed to help you understand the key concepts. The goal is to provide explicit implementations of popular layers, models, and optimizers. Above all else, this code is designed to be readable and clear. Many of these examples are modified from Stanford's CS 230 / 231N course materials available online.

## EduTorch vs PyTorch

One notable difference between EduTorch and PyTorch is that EduTorch _does NOT provide autograd_. There are many educational benefits to deriving/implementing the backprop step yourself, and if you want automatic gradient calculations, you are better off using the real framework. If you really want autograd for an EduTorch-like project, you might want to consider using Brown University's [BrunoFlow](https://github.com/Brown-Deep-Learning/brunoflow). Or, if you just want to learn how the autograd system is implemented, you can check out Andrej Karpathy's [micrograd project](https://github.com/karpathy/micrograd).

There is no CUDA or GPU support for EduTorch either, for the same reasons.

# Contributing

All issues and pull requests are much appreciated!

- First, be sure to run `pre-commit install`.
- To run all tests and use auto-formatting tools, use `pre-commit run`.
- To only run unit tests, run `pytest`.
