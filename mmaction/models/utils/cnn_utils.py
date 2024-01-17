import torch
from torch import nn


class Conv3d_1xnxn(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N*T, C, H, W)
        x = self._conv_forward(x, self.weight, self.bias)
        _, C, H, W = x.shape # 卷积操作可能会改变x的形状，比如下采样会让宽高减半
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return x


def dwconv_1xnxn(channels, kernel_size, stride=1, padding=0, dilation=1):
    return Conv3d_1xnxn(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=channels)

def conv_1xnxn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return Conv3d_1xnxn(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

# def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)

# def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
#     return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

# def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
#     return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

# def conv_1x1x1(inp, oup, groups=1):
#     return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

# def conv_3x3x3(inp, oup, groups=1):
#     return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

# def conv_5x5x5(inp, oup, groups=1):
#     return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)
