# Pytorch Toolkit

This package is to facilitate fast and reusable development of research projects.

## Installation

`pip install torkit`

## Important Pytorch Updates

### 1.7.0

- [torch.save seems to be fixed](https://github.com/pytorch/pytorch/pull/46207).

### 1.6.0

- [torch.save might save wrong weights occasionally](https://github.com/pytorch/pytorch/issues/46020).

### 1.5.1

- torch.svd is wrong when the matrix is an orthonormal matrix with -1 as determinant.
- [Fix memory usage increase in 1.5.0](https://github.com/pytorch/pytorch/pull/38674).

### 1.4.0

- [Learning rate schedulers become chainable](https://github.com/pytorch/pytorch/pull/26423). Thus, `get_lr` is changed
  to `get_last_lr`.
