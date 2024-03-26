import torch

def window_partition_0(x, window_size):
    N, B, D, H, W, C = x.shape
    windows = x.view(N, B, D // (window_size[0]), window_size[0], H // (window_size[1]), window_size[1], W // (window_size[2]), window_size[2], C)
    windows = windows.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous()
    return windows.view(N, -1, window_size[0]*window_size[1]*window_size[2], C)

def window_partition_2(x, window_size):
    N, B, D, H, W, C = x.shape
    x = x.view(N, B, D // 2, 2, H // 2, 2, W // 2, 2, C)
    windows = x.view(N, B, D // (2*window_size[0]), window_size[0], 2, H // (2*window_size[1]), window_size[1], 2, W // (2*window_size[2]), window_size[2], 2, C)
    windows = windows.permute(0, 1, 2, 4, 5, 7, 8, 10, 3, 6, 9, 11).contiguous()
    return windows.view(N, -1, window_size[0]*window_size[1]*window_size[2], C)

def window_partition_4(x, window_size):
    N, B, D, H, W, C = x.shape
    x = x.view(N, B, D // 4, 4, H // 4, 4, W // 4, 4, C)
    windows = x.view(N, B, D // (4*window_size[0]), window_size[0], 4, H // (4*window_size[1]), window_size[1], 4, W // (4*window_size[2]), window_size[2], 4, C)
    windows = windows.permute(0, 1, 2, 4, 5, 7, 8, 10, 3, 6, 9, 11).contiguous()
    return windows.view(N, -1, window_size[0]*window_size[1]*window_size[2], C)

def window_partition_8(x, window_size):
    N, B, D, H, W, C = x.shape
    x = x.view(N, B, D // 8, 8, H // 8, 8, W // 8, 8, C)
    windows = x.view(N, B, D // (8*window_size[0]), window_size[0], 8, H // (8*window_size[1]), window_size[1], 8, W // (8*window_size[2]), window_size[2], 8, C)
    windows = windows.permute(0, 1, 2, 4, 5, 7, 8, 10, 3, 6, 9, 11).contiguous()
    return windows.view(N, -1, window_size[0]*window_size[1]*window_size[2], C)

def ASPAwindow_partition(QKV, window_size, idx):
    """
    For window division, for the Atrous Spatial Pyramid Windows,
    the expansion rate of the first and second layers is (2, 4, 8),
    the third layer is (2, 4), and the fourth layer is (2).

    Args:
        QKV: (3, B, D, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    N, B, D, H, W, C = QKV.shape

    if idx == 0:
        QKV_win = window_partition_0(QKV, window_size).unsqueeze(1)
    elif idx == 2:
        QKV_win = window_partition_2(QKV, window_size).unsqueeze(1)
    elif idx == 4:
        qkv1 = QKV[:, :, :, :, :, :C // 2]
        qkv2 = QKV[:, :, :, :, :, C // 2:]
        qkv1_win = window_partition_2(qkv1, window_size).unsqueeze(1)
        qkv2_win = window_partition_4(qkv2, window_size).unsqueeze(1)
        QKV_win = [qkv1_win, qkv2_win]
        QKV_win = torch.cat(QKV_win, dim=1)
    else:
        qkv1 = QKV[:, :, :, :, :, :C // 3]
        qkv2 = QKV[:, :, :, :, :, C // 3:2 * C // 3]
        qkv3 = QKV[:, :, :, :, :, 2 * C // 3:]
        qkv1_win = window_partition_2(qkv1, window_size).unsqueeze(1)
        qkv2_win = window_partition_4(qkv2, window_size).unsqueeze(1)
        qkv3_win = window_partition_8(qkv3, window_size).unsqueeze(1)
        QKV_win = [qkv1_win, qkv2_win, qkv3_win]
        QKV_win = torch.cat(QKV_win, dim=1)

    return QKV_win


def window_reverse_0(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def window_reverse_2(windows, window_size, B, D, H, W):
    x = windows.view(B, D // (2*window_size[0]), 2, H // (2*window_size[1]), 2, W // (2*window_size[2]), 2, window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 7, 2, 3, 8, 4, 5, 9, 6, 10).contiguous().view(B, D, H, W, -1)
    return x

def window_reverse_4(windows, window_size, B, D, H, W):
    x = windows.view(B, D // (4*window_size[0]), 4, H // (4*window_size[1]), 4, W // (4*window_size[2]), 4, window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 7, 2, 3, 8, 4, 5, 9, 6, 10).contiguous().view(B, D, H, W, -1)
    return x

def window_reverse_8(windows, window_size, B, D, H, W):
    x = windows.view(B, D // (8*window_size[0]), 8, H // (8*window_size[1]), 8, W // (8*window_size[2]), 8, window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 7, 2, 3, 8, 4, 5, 9, 6, 10).contiguous().view(B, D, H, W, -1)
    return x

def ASPAwindow_reverse(windows, window_size, input_resolution, idx):
    """
    For window recovery

    Args:
        windows: (r, num_windows*B, window_size*window_size, C)
        window_size (int): Window size
        input_resolution: input resolution (D, H, W)

    Returns:
        x: (B, D, H, W, C)
    """
    D = input_resolution[0]
    H = input_resolution[1]
    W = input_resolution[2]
    r, B_, N, C = windows.size()
    windows = windows.view(r, B_, window_size[0], window_size[1], window_size[2], C)
    B = int(B_ / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    if idx == 0:
        x = window_reverse_0(windows.squeeze(0), window_size, B, D, H, W)
    elif idx == 2:
        x = window_reverse_2(windows.squeeze(0), window_size, B, D, H, W)
    elif idx == 4:
        windows1, windows2 = windows.unbind(0)
        x1 = window_reverse_2(windows1, window_size, B, D, H, W)
        x2 = window_reverse_4(windows2, window_size, B, D, H, W)
        x = torch.cat([x1, x2], dim=-1)
    else:
        windows1, windows2, windows3 = windows.unbind(0)
        x1 = window_reverse_2(windows1, window_size, B, D, H, W)
        x2 = window_reverse_4(windows2, window_size, B, D, H, W)
        x3 = window_reverse_8(windows3, window_size, B, D, H, W)
        x = torch.cat([x1, x2, x3], dim=-1)

    return x