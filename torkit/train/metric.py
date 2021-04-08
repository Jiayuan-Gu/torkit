"""Metric(scalar)"""

from __future__ import division
from collections import deque
from typing import Union

import numpy as np
import torch

__all__ = ['Metric', 'Average', 'Accuracy']


class Metric(object):
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @property
    def result(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    @property
    def summary_str(self):
        raise NotImplementedError()


class Average(Metric):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    default_fmt = '{avg:.4f} ({global_avg:.4f})'
    default_summary_fmt = '{global_avg:.4f}'

    def __init__(self, window_size=20, fmt=None, summary_fmt=None):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0
        self.fmt = fmt or self.default_fmt
        self.summary_fmt = summary_fmt or self.default_summary_fmt

    def update(self, value: Union[torch.Tensor, float], count=1):
        self.values.append(value.item() if torch.is_tensor(value) else value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    def reset(self):
        self.values.clear()
        self.counts.clear()
        self.sum = 0.0
        self.count = 0

    @property
    def result(self):
        return self.global_avg

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)

    @property
    def summary_str(self):
        return self.summary_fmt.format(global_avg=self.global_avg)

    @property
    def avg(self):
        counts = np.sum(self.counts)
        return np.sum(self.values) / counts if counts != 0 else float('nan')

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else float('nan')


class Accuracy(Average):
    default_fmt = '{avg:.2f} ({global_avg:.2f})'
    default_summary_fmt = '{global_avg:.2f}'

    def update(self, y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray]):
        assert y_pred.shape == y_true.shape, 'Mismatched shapes: y_pred({}) vs y_true({}).'.format(
            y_pred.shape, y_true.shape
        )
        if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
            mask = torch.eq(y_pred, y_true)
            value = mask.float().sum().item()
            count = mask.numel()
        elif isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            mask = np.equal(y_pred, y_true)
            value = mask.sum().item()
            count = mask.size
        else:
            raise TypeError('{}, {}'.format(type(y_pred), type(y_true)))
        super().update(value=value, count=count)

    @property
    def avg(self):
        return super().avg * 100.0

    @property
    def global_avg(self):
        return super().global_avg * 100.0


class TestAccuracy:
    y_pred = np.array([1, 0, 1])
    y_true = np.array([1, 0, 0])
    expected = 2.0 / 3.0 * 100.0

    def test_np(self):
        metric = Accuracy()
        metric.update(self.y_pred, self.y_true)
        np.testing.assert_allclose(metric.result, self.expected)

    def test_tensor(self):
        metric = Accuracy()
        metric.update(torch.tensor(self.y_pred), torch.tensor(self.y_true))
        np.testing.assert_allclose(metric.result, self.expected)
