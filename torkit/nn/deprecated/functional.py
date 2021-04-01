import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
# Distance
# ---------------------------------------------------------------------------- #
def bpdist(feature, data_format='BCN'):
    """Compute pairwise (square) distances of features.
    Based on $(x-y)^2=x^2+y^2-2xy$.

    Args:
        feature (torch.Tensor): [B, C, N] or [B, N, C]
        data_format (str): the format of features. [BCN/BNC]

    Returns:
        torch.Tensor: (B, N, N)

    Notes:
        The function can be replaced by torch.cdist.
        This method returns square distances, and is optimized for lower memory and faster speed.
        Square sum is more efficient than gather diagonal from inner product.
        The result is somehow inaccurate compared to directly using $(x-y)^2$.
    """
    if data_format == 'BCN':
        square_sum = torch.sum(feature ** 2, 1, keepdim=True)
        square_sum = square_sum.transpose(1, 2) + square_sum
        distance = torch.baddbmm(square_sum, feature.transpose(1, 2), feature, alpha=-2.0)
    elif data_format == 'BNC':
        square_sum = torch.sum(feature ** 2, 2, keepdim=True)
        square_sum = square_sum.transpose(1, 2) + square_sum
        distance = torch.baddbmm(square_sum, feature, feature.transpose(1, 2), alpha=-2.0)
    else:
        raise RuntimeError('Unsupported data format {}.'.format(data_format))
    return distance


def bpdist2(feature1, feature2, data_format='BCN'):
    """Compute pairwise (square) distances of two features.

    Args:
        feature1 (torch.Tensor): [B, C, N1]
        feature2 (torch.Tensor): [B, C, N2]
        data_format (str): the format of features. [BCN/BNC]

    Returns:
        torch.Tensor: [B, N1, N2]

    Notes:
        The function can be replaced by torch.cdist.
    """
    if data_format == 'BCN':
        square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
        square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
        square_sum = square_sum1.transpose(1, 2) + square_sum2
        distance = torch.baddbmm(square_sum, feature1.transpose(1, 2), feature2, alpha=-2.0)
    elif data_format == 'BNC':
        square_sum1 = torch.sum(feature1 ** 2, 2, keepdim=True)
        square_sum2 = torch.sum(feature2 ** 2, 2, keepdim=True)
        square_sum = square_sum1 + square_sum2.transpose(1, 2)
        distance = torch.baddbmm(square_sum, feature1, feature2.transpose(1, 2), alpha=-2.0)
    else:
        raise RuntimeError('Unsupported data format {}.'.format(data_format))
    return distance


# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
def encode_one_hot(target, num_classes):
    """Encode integer labels into one-hot vectors.

    Args:
        target (torch.Tensor): [N]
        num_classes (int): the number of classes

    Returns:
        torch.FloatTensor: [N, C]

    Notes:
        The function can be replaced by torch.nn.functional.one_hot.
    """
    one_hot = target.new_zeros(target.size(0), num_classes)
    one_hot = one_hot.scatter(1, target.unsqueeze(1), 1)
    return one_hot.float()
