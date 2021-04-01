import torch
from torch import nn
from .linear import LinearBNReLU
from .conv import Conv1dBNReLU

__all__ = ['mlp_bn_relu', 'mlp1d_bn_relu']


def mlp_bn_relu(in_channels, out_channels_list, bn=True):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(LinearBNReLU(c_in, c_out, relu=True, bn=bn))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp1d_bn_relu(in_channels, out_channels_list, bn=True):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv1dBNReLU(c_in, c_out, 1, relu=True, bn=bn))
        c_in = c_out
    return nn.Sequential(*layers)
