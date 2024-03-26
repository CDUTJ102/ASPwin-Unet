import math
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DLA(nn.Module):
    def __init__(self, inp, oup, kernel_size = 3, stride=1, expand_ratio = 3, refine_mode='none'):
        super(DLA, self).__init__()
        """
            Distributed Local Attention used for refining the attention map.
        """

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp, self.oup = inp, oup
        self.high_dim_id = False
        self.refine_mode = refine_mode


        if refine_mode == 'conv':
            self.conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size,kernel_size), stride, (1,1), groups=1, bias=False)
        elif refine_mode == 'conv_exapnd':
            if self.expand_ratio != 1:
                self.conv_exp = Conv2dSamePadding(inp, hidden_dim, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.depth_sep_conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size,kernel_size), stride, (1,1), groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)

            self.conv_pro = Conv2dSamePadding(hidden_dim, oup, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

            self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x = input
        if self.refine_mode == 'conv':
            return self.conv(x)
        else:
            if self.expand_ratio !=1:
                x = self.relu(self.bn1(self.conv_exp(x)))
            x = self.relu(self.bn2(self.depth_sep_conv(x)))
            x = self.bn3(self.conv_pro(x))
            if self.identity:
                return x + input
            else:
                return x