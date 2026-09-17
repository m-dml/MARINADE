import torch


def sigmoid_mask(x, a, b, sharpness=16.0):
    """Returns :math:`\\sigma(sharpness x) a + (1 - \\sigma(sharpness x)) b`"""
    w = torch.sigmoid(sharpness * x)
    return w * a + (1 - w) * b


def smooth_strict_mask(x, a, b, sharpness=1024.0):
    """Mask that is strictly :math:`b` for :math:`x < 0` and smoothly approaches :math:`a` for :math:`x > 0`"""
    w = torch.zeros_like(x)
    pos = x > 0
    w[pos] = torch.exp(-(1.0 / sharpness) / x[pos])
    return w * a + (1 - w) * b
