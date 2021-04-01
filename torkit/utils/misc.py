"""Miscellaneous functions"""

import numpy as np
import torch


def print_dict(d: dict):
    """Print the given dictionary for debugging."""
    for k, v in d.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)
