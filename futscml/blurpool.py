"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`
Hacked together by Chris Ha and Ross Wightman
"""
from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling
    Args:
        channels = Number of input channels
        kernel_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride
    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(self, channels, kernel_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert kernel_size > 1
        self.channels = channels
        self.filt_size = kernel_size
        self.stride = stride
        self.padding = [get_padding(kernel_size, stride, dilation=1)] * 4
        coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].repeat(self.channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, 'reflect')
        return F.conv2d(x, self.filt, stride=self.stride, groups=x.shape[1])