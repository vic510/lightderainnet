"""PSNR / SSIM。

預設採去雨文獻常見做法：輸出先量化成 8-bit，再於 YCbCr 的 Y 通道計算。
報告時務必註明使用的設定（Y 或 RGB、是否裁邊），不同設定的數字不能直接比較。
"""
import torch
import torch.nn.functional as F


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """BCHW、RGB、範圍 [0,1] -> B1HW 的 Y 通道（ITU-R BT.601，與 MATLAB rgb2ycbcr 相同）。"""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def quantize(img: torch.Tensor) -> torch.Tensor:
    return torch.round(img.clamp(0, 1) * 255.0) / 255.0


def _prepare(img, y_channel, crop):
    img = img.double()
    if y_channel:
        img = rgb_to_y(img)
    if crop > 0:
        img = img[..., crop:-crop, crop:-crop]
    return img


def psnr(pred, target, y_channel=True, crop=0):
    """回傳每張影像的 PSNR (dB)，shape [B]。"""
    p, t = _prepare(pred, y_channel, crop), _prepare(target, y_channel, crop)
    mse = ((p - t) ** 2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def _gaussian_window(size, sigma, channels, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim_map(p, t, window=11, sigma=1.5):
    """Wang et al. 2004 的 SSIM（高斯視窗 11、sigma 1.5、K1=0.01、K2=0.03、資料範圍 1）。"""
    c = p.shape[1]
    w = _gaussian_window(window, sigma, c, p.device, p.dtype)
    conv = lambda x: F.conv2d(x, w, groups=c)  # noqa: E731
    mu_p, mu_t = conv(p), conv(t)
    var_p = conv(p * p) - mu_p ** 2
    var_t = conv(t * t) - mu_t ** 2
    cov = conv(p * t) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))


def ssim(pred, target, y_channel=True, crop=0):
    """回傳每張影像的 SSIM，shape [B]。"""
    p, t = _prepare(pred, y_channel, crop), _prepare(target, y_channel, crop)
    return ssim_map(p, t).flatten(1).mean(1)


def ssim_loss(pred, target):
    """訓練用：RGB、float32、可微分。"""
    return 1.0 - ssim_map(pred, target).mean()
