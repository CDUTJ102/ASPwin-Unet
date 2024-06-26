from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from ASPwin_unet import BasicLayer, BasicLayerUp


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(OrderedDict([
            ('conv3d', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)),
            ('gelu', nn.GELU()),
            ('InstanceNorm', nn.InstanceNorm3d(out_channels, affine=True)),
        ]))
        # self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.single_conv(x)

        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=(2, 4, 4)):
        super(UpConv, self).__init__()
        self.scale_factor = scale_factor

        self.upsampling = nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=scale_factor,
                                             stride=scale_factor)

    def forward(self, x):
        x = self.upsampling(x)
        return x


class PatchExpanding(nn.Module):
    """
    Patch Expanding layer for up-sampling
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.expanding = nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, D % 2, 0, W % 2, 0, H % 2))

        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.expanding(x)
        x = rearrange(x, 'b c d h w -> b d h w c')

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer for down-sampling

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, mode='default', norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        assert mode in [None, 'default', 'max-pooling', 'avg-pooling', 'interpolation', 'conv']
        self.mode = mode
        self.norm = nn.Identity()
        self.reduction = nn.Identity()
        self.act = nn.Identity()

        if self.mode == 'default':
            self.norm = norm_layer(8 * dim)
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        elif self.mode == 'max-pooling':
            self.act = nn.GELU()
            self.norm = norm_layer(dim)
            self.reduction = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(dim, 2 * dim, kernel_size=3, stride=1, padding=1)
            )
        elif self.mode == 'avg-pooling':
            self.act = nn.GELU()
            self.norm = norm_layer(dim)
            self.reduction = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(dim, 2 * dim, kernel_size=3, stride=1, padding=1)
            )
        elif self.mode == 'conv':
            self.act = nn.GELU()
            self.norm = norm_layer(dim)
            self.reduction = nn.Conv3d(dim, 2 * dim, kernel_size=3, stride=2, padding=1)

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

        if self.mode == 'default':
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
            x1 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
            x2 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
            x3 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
            x4 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
            x5 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
            x6 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
            x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
            x = self.norm(x)  # increased C, then norm
            x = self.reduction(x)
        else:
            x = self.act(x)
            x = self.norm(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x = self.reduction(x)  # b c d/2 h/2 w/2
            x = rearrange(x, 'b c d h w -> b d h w c')

        return x


class PatchConv3D(nn.Module):
    """
    replace PatchEmbed3D by Conv3d stem to to increase optimization stability.

    Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollár, P., & Girshick, R. (2021).
    Early Convolutions Help Transformers See Better. 1–15. https://arxiv.org/abs/2106.14881
    """

    def __init__(self, patch_size=(2, 2, 2), in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(OrderedDict([
            ('downsampling_1', SingleConv(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1)),
            ('conv3x3_1', SingleConv(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1)),
            ('downsampling_2', SingleConv(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1)),
            ('conv3x3_2', nn.Conv3d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1))
        ]))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_channels (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        overlapping (bool): generate patches by overlapping mode. Default: False
    """

    def __init__(self, patch_size=(4, 4, 4), in_channels=3, embed_dim=96, norm_layer=None, overlapping=False):
        super().__init__()
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if overlapping:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=2 * np.array(patch_size) - 1,
                                  stride=patch_size,
                                  padding=np.array(patch_size) // 2)
        else:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class ASPwinUnet3D(nn.Module):
    """
    modified from ASPwin Transformer

    Args:
        initialized (None | str): how parameters initialized. Default: None
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        conv_stem (bool): whether use convolutional stem for patch embedding. Default: False
        overlapped_embed (bool): whether embed by a overlapped conv. Default: False
        downsample_mode (str): how features down-sampled. Default: 'default' -> Swin transformer like
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each ASPwin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate after patch embedding.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 patch_size=(2, 4, 4),
                 conv_stem=False,
                 overlapped_embed=False,
                 downsample_mode='default',  # ('default', 'max-pooling', 'avg-pooling', 'interpolation')
                 in_channels=3,
                 embed_dim=96,
                 depths=(2, 2, 2, 1),
                 num_heads=(3, 6, 12, 24),
                 window_size=(4, 4, 4),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 patch_norm=False,
                 out_indices=(0, 1, 2),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 norm_layer=nn.LayerNorm, ):
        super().__init__()

        self.pretrained = pretrained
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices if len(out_indices) == len(depths) - 1 else [i for i in range(len(depths) - 1)]
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.conv_stem = conv_stem
        self.downsample_mode = downsample_mode
        self.overlapped_embed = overlapped_embed
        patches_resolution = [128 // patch_size[0], 128 // patch_size[1], 128 // patch_size[2]]

        if conv_stem:
            self.patch_embed = PatchConv3D(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
                                           norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
                                            norm_layer=norm_layer if self.patch_norm else None,
                                            overlapping=overlapped_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoders
        self.encoders = nn.ModuleList()
        for i_encoder in range(self.num_layers - 1):  # (0, 1, 2)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_encoder),
                input_resolution=[patches_resolution[0] // (2 ** i_encoder),
                                  patches_resolution[1] // (2 ** i_encoder),
                                  patches_resolution[2] // (2 ** i_encoder)],
                depth=depths[i_encoder],  # (2, 2, 2)
                num_heads=num_heads[i_encoder],  # (3, 6, 12)
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_encoder]):sum(depths[:i_encoder + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                downsample_mode=self.downsample_mode,
                use_checkpoint=use_checkpoint)
            self.encoders.append(layer)

        # build decoders
        self.decoders = nn.ModuleList()
        for i_decoder in range(self.num_layers):  # (0, 1, 2, 3)
            decoder = BasicLayerUp(
                dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_decoder)),
                input_resolution=[patches_resolution[0] // (2 ** (self.num_layers - 1 - i_decoder)),
                                  patches_resolution[1] // (2 ** (self.num_layers - 1 - i_decoder)),
                                  patches_resolution[2] // (2 ** (self.num_layers - 1 - i_decoder))],
                depth=depths[self.num_layers - 1 - i_decoder],  # (2, 2, 2, 4)
                num_heads=num_heads[self.num_layers - 1 - i_decoder],  # (3, 6, 12, 24)
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:self.num_layers - 1 - i_decoder]):sum(depths[:self.num_layers - i_decoder])],
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_decoder < 3 else None,
                use_checkpoint=use_checkpoint)
            self.decoders.append(decoder)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]  # [96, 196, 384, 768]
        self.num_features = num_features

        # fusion layers, optimization needed
        self.fusions = nn.ModuleList()
        for i in range(self.num_layers - 1):  # [0, 1, 2]
            fusion = nn.Sequential(
                # nn.Conv3d(in_channels=2 * num_features[i], out_channels=2 * num_features[i], kernel_size=3, stride=1,
                #           padding=1, groups=2 * num_features[i]),
                # nn.GELU(),
                # nn.InstanceNorm3d(2 * num_features[i], affine=True),
                nn.Conv3d(in_channels=2 * num_features[i], out_channels=num_features[i], kernel_size=1),
                nn.GELU()
                )
            self.fusions.append(fusion)
        self.fusions = self.fusions[::-1]

        # add a norm layer for each output. remove or move to basic layer
        for i_layer in self.out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            act = nn.GELU()
            act_name = f'act{i_layer}'
            self.add_module(act_name, act)

        # for fine tuning, default not used
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.encoders[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)  # B C D Wh Ww

        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers - 1):
            layer = self.encoders[i]
            # print('encoder{}: {}'.format(i, type(layer)))
            # print(layer)
            x_out, x = layer(x)  # x is the output after down-sampling, x_out is output without down-sampling

            if i in self.out_indices:  # remove or move to basic layer
                x_out = rearrange(x_out, 'b c d h w -> b d h w c')
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = rearrange(x_out, 'b d h w c -> b c d h w')
                # print('encoder{}: {}'.format(i, out.shape))
                outs.append(out)

        for i in range(self.num_layers):  # [0, 1, 2, 3]
            decoder = self.decoders[i]
            x_out, x = decoder(x)  # x means x_up except the last one
            # skip connection, prepare y for next stage
            if i < self.num_layers - 1:
                shortcut = outs.pop()
                x = torch.cat([x, shortcut], dim=1)  # skip connection
                x = self.fusions[i](x)  # fusion
            # print('decoder{}: {}'.format(i, x.shape))

        return x


class DoseformerV2(nn.Module):
    def __init__(self, configs):
        super(DoseformerV2, self).__init__()
        self.ASPwin_unet3d = ASPwinUnet3D(*configs['model']['ASPwinUnet3D'].values(),
                                      norm_layer=torch.nn.LayerNorm)

        # [96, 196, 384, 768]
        num_layers = len(configs['model']['ASPwinUnet3D']['depths'])
        num_features = [int(configs['model']['ASPwinUnet3D']['embed_dim'] * 2 ** i) for i in range(num_layers)]
        embed_dim = int(configs['model']['ASPwinUnet3D']['embed_dim'])

        # head needed
        self.upconv = UpConv(in_ch=num_features[0], out_ch=embed_dim, scale_factor=configs['model']['ASPwinUnet3D']['patch_size'])
        self.contour_dwconv = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)
        self.contour_conv = nn.Conv3d(in_channels=embed_dim, out_channels=1, kernel_size=3, padding=1)
        self.x_conv = nn.Conv3d(in_channels=embed_dim, out_channels=1, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.nor1 = nn.InstanceNorm3d(embed_dim, affine=True)
        self.nor2 = nn.InstanceNorm3d(1, affine=True)
        self.final_conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.final_conv = nn.Conv3d(in_channels=6, out_channels=1, kernel_size=1)


    def head(self, x):
        up_x = self.upconv(x)

        # contour
        contour = self.contour_dwconv(up_x)
        contour = self.act(contour)
        contour = self.contour_conv(contour)

        # dose
        # x_1 = self.x_dwconv1(up_x)
        # x_1 = self.act(x_1)
        # x_1 = self.nor1(x_1)
        x_ = self.x_conv(up_x)
        x_ = self.act(x_)
        x_ = self.nor2(x_)

        x = torch.cat([x_, contour.detach()], dim=1)  # Contour information is included, but the gradient is truncated
        x = self.final_conv(x)
        pred = x + x_

        return contour, pred


    def forward(self, x):
        x = self.ASPwin_unet3d(x)
        contour, pred = self.head(x)

        return contour, pred

