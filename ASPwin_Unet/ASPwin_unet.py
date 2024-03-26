from functools import reduce, lru_cache
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from ASP_windows_process import ASPAwindow_partition, ASPAwindow_reverse
from DLA import DLA


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=1, groups=in_features)
        self.norm = nn.InstanceNorm3d(in_features, affine=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, idx, dim, input_resolution, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., focusing_factor=3, kernel_size=5):

        super().__init__()
        self.idx = idx
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focusing_factor = focusing_factor

        if self.idx == 1:
            self.num_heads_rate = 0
            self.rate = 1
        else:
            if min(self.input_resolution) // max(window_size) == 16:
                self.num_heads_rate = 8
                self.rate = 3
            elif min(self.input_resolution) // max(window_size) == 8:
                self.num_heads_rate = 8
                self.rate = 3
            elif min(self.input_resolution) // max(window_size) == 4:
                self.num_heads_rate = 4
                self.rate = 2
            else:
                self.num_heads_rate = 2
                self.rate = 1

        # if min(self.input_resolution) // max(window_size) == 16:
        #     self.qkv = nn.Linear(dim, dim * 3 * self.rate, bias=qkv_bias)
        #     self.num_heads = self.num_heads * self.rate
        #     self.scale = nn.Parameter(torch.zeros(size=(self.rate, 1, 1, dim * self.rate // self.num_heads)))
        #     self.positional_encoding = nn.Parameter(
        #         torch.zeros(size=(self.rate, 1, window_size[0] * window_size[1] * window_size[2], dim * self.rate // self.num_heads)))
        #     self.proj = nn.Linear(dim * self.rate, dim)
        # else:
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Focused Linear Attention
        self.scale = nn.Parameter(torch.zeros(size=(self.rate, 1, 1, dim // self.num_heads)))
        self.positional_encoding = nn.Parameter(
            torch.zeros(size=(self.rate, 1, window_size[0] * window_size[1] * window_size[2], dim // self.num_heads)))
        self.proj = nn.Linear(dim, dim)

        self.dwc = nn.Conv3d(in_channels=head_dim * self.rate, out_channels=head_dim * self.rate,
                             kernel_size=kernel_size, groups=head_dim * self.rate, padding=kernel_size // 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.positional_encoding, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, D, H, W, C)
        """

        B, D, H, W, C = x.shape

        # if min(self.input_resolution) // max(self.window_size) == 16:
        #     QKV = self.qkv(x).reshape(B, D, H, W, 3, C * self.rate).permute(4, 0, 1, 2, 3, 5)
        # else:
        QKV = self.qkv(x).reshape(B, D, H, W, 3, C).permute(4, 0, 1, 2, 3, 5)
        QKV_win = ASPAwindow_partition(QKV, self.window_size, self.num_heads_rate)  # 3, r, B_, Wd*Wh*Ww, C
        q, k, v = QKV_win.unbind(0)

        q, k, v = (rearrange(x, "r b n (h c) -> r (b h) n c", h=self.num_heads // self.rate) for x in [q, k, v])
        k = k + self.positional_encoding.to('cuda')
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale).to('cuda')
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("r b i c, r b c -> r b i", q, k.sum(dim=2)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("r b j c, r b j d -> r b c d", k, v)
            x = torch.einsum("r b i c, r b c d, r b i -> r b i d", q, kv, z)
        else:
            qk = torch.einsum("r b i c, r b j c -> r b i j", q, k)
            # DLA
            # if min(self.input_resolution) == 32:
            #     qk = rearrange(qk, "r (b h) i j -> b (r h) i j", h=self.num_heads // self.rate)
            #     qk = self.adapt_bn(self.DLA(qk))
            #     qk = rearrange(qk, "b (r h) i j -> r (b h) i j", h=self.num_heads // self.rate)

            x = torch.einsum("r b i j, r b j d, r b i -> r b i d", qk, v, z)

        feature_map = rearrange(v, "r b (d w h) c -> b (r c) d w h", d=self.window_size[0], w=self.window_size[0], h=self.window_size[0])
        feature_map = rearrange(self.dwc(feature_map), "b (r c) d w h -> r b (d w h) c", r=self.rate)
        x = x + feature_map
        x = rearrange(x, "r (b h) n c -> r b n (h c)", h=self.num_heads // self.rate)
        x = ASPAwindow_reverse(x, self.window_size, self.input_resolution, self.num_heads_rate)  # (B, D, H, W, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ASPwinTransformerBlock3D(nn.Module):
    """ ASPwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, idx, dim, input_resolution, num_heads, window_size=(2, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            idx=idx, dim=dim, input_resolution=input_resolution, window_size=self.window_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        x = self.norm1(x)
        x = self.attn(x)

        return x

    def forward_part2(self, x):
        shortcut = x
        return shortcut + self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic ASPwin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (4,4,4).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 downsample_mode=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ASPwinTransformerBlock3D(
                idx=0 if i % 2 == 0 else 1,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, mode=downsample_mode, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for W-MSA
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')

        for blk in self.blocks:
            x = blk(x)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x_down = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x_down = rearrange(x_down, 'b d h w c -> b c d h w')
            return x, x_down
        else:
            x = rearrange(x, 'b d h w c -> b c d h w')
            return x, x


class BasicLayerUp(nn.Module):
    """ A basic ASPwin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (4,4,4).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ASPwinTransformerBlock3D(
                idx=0 if i % 2 == 0 else 1,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for ASPW-MSA
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')

        for blk in self.blocks:
            x = blk(x)
        x = x.view(B, D, H, W, -1)

        if self.upsample is not None:
            x_up = self.upsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x_up = rearrange(x_up, 'b d h w c -> b c d h w')
            return x, x_up
        else:
            x = rearrange(x, 'b d h w c -> b c d h w')
            return x, x

