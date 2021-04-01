import numpy as np
import scipy.spatial.distance as sdist
import torch

from .functional import bpdist, bpdist2


def test_bpdist():
    batch_size = 16
    channels = 64
    num_inst = 1024

    feature_np = np.random.rand(batch_size, channels, num_inst)
    feature_tensor = torch.from_numpy(feature_np)
    if torch.cuda.is_available():
        feature_tensor = feature_tensor.cuda()

    # check pairwise distance
    distance_np = np.stack([sdist.squareform(np.square(sdist.pdist(x.T))) for x in feature_np])
    distance_tensor = bpdist(feature_tensor)
    np.testing.assert_allclose(distance_np, distance_tensor.cpu().numpy(), atol=1e-6)

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     bpdist(feature_tensor)
    # print(prof)
    # print(torch.cuda.max_memory_allocated() / (1024.0 ** 2))


def test_bpdist2():
    batch_size = 16
    channels = 64
    num_inst1 = 1023
    num_inst2 = 1025

    feature1_np = np.random.rand(batch_size, channels, num_inst1)
    feature2_np = np.random.rand(batch_size, channels, num_inst2)
    feature1_tensor = torch.from_numpy(feature1_np)
    feature2_tensor = torch.from_numpy(feature2_np)
    if torch.cuda.is_available():
        feature1_tensor = feature1_tensor.cuda()
        feature2_tensor = feature2_tensor.cuda()

    # check pairwise distance_np
    distance_np = np.stack([np.square(sdist.cdist(x.T, y.T)) for x, y in zip(feature1_np, feature2_np)])
    distance_tensor = bpdist2(feature1_tensor, feature2_tensor)  # warm up
    np.testing.assert_allclose(distance_np, distance_tensor.cpu().numpy())

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     bpdist2(feature1_tensor, feature2_tensor)
    # print(prof)
