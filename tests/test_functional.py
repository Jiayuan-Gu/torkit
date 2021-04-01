import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F

from torkit.nn.functional import smooth_cross_entropy
from torkit.nn.functional import batch_index_select


def test_smooth_cross_entropy():
    num_samples = 2
    num_classes = 10
    label_smoothing = 0.1

    # numpy version
    target_np = np.random.randint(0, num_classes, [num_samples])
    one_hot_np = np.zeros([num_samples, num_classes])
    one_hot_np[np.arange(num_samples), target_np] = 1.0
    smooth_one_hot = one_hot_np * (1.0 - label_smoothing) + np.ones_like(one_hot_np) * label_smoothing / num_classes
    logit_np = np.random.randn(num_samples, num_classes)
    prob_np = softmax(logit_np, axis=-1)
    cross_entropy_np = - (smooth_one_hot * np.log(prob_np)).sum(1).mean()

    # torch version
    target_tensor = torch.from_numpy(target_np)
    logit_tensor = torch.from_numpy(logit_np)
    one_hot_tensor = F.one_hot(target_tensor, num_classes)
    np.testing.assert_allclose(one_hot_np, one_hot_tensor.numpy())
    cross_entropy_tensor = smooth_cross_entropy(logit_tensor, target_tensor, label_smoothing)
    np.testing.assert_allclose(cross_entropy_np, cross_entropy_tensor.numpy())


def test_batch_index_select():
    shape = (2, 16, 9, 32)
    batch_size = shape[0]
    input_np = np.random.randn(*shape)

    for dim in range(1, len(shape)):
        num_select = np.random.randint(shape[dim])
        index_np = np.random.randint(shape[dim], size=(batch_size, num_select))
        target_np = np.stack([np.take(input_np[b], index_np[b], axis=dim - 1) for b in range(batch_size)], axis=0)

        input_tensor = torch.tensor(input_np)
        index_tensor = torch.tensor(index_np)
        target_tensor = batch_index_select(input_tensor, index_tensor, dim=dim)
        np.testing.assert_allclose(target_np, target_tensor.numpy())
