# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
from __future__ import division
from collections import defaultdict

import numpy as np
import torch
from ..train.metric import Average


class MetricLogger(object):
    """Metric logger"""

    def __init__(self, delimiter='\t'):
        self.metrics = defaultdict(Average)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                count = v.numel()
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, np.ndarray):
                count = v.size
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, (tuple, list)):
                value, count = v
                value = value.item()
                count = count.item()
            elif isinstance(v, (float, int)):
                value = v
                count = 1
            else:
                raise TypeError('Unsupported type: '.format(type(v)))
            self.metrics[k].update(value, count)

    def __getitem__(self, item):
        return self.metrics[item]

    def __str__(self):
        ret_str = []
        for name, metric in self.metrics.items():
            ret_str.append('{}: {}'.format(name, str(metric)))
        return self.delimiter.join(ret_str)

    @property
    def summary_str(self):
        ret_str = []
        for name, metric in self.metrics.items():
            ret_str.append('{}: {}'.format(name, metric.summary_str))
        return self.delimiter.join(ret_str)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
