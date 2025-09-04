# 代码内容：
"""
作者： hsjxg
日期： 2025/8/5 18:15
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from torchvision.transforms.functional import rgb_to_grayscale


def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(pred, target, window_size=11):
    """
    Compute SSIM between two images (batch-wise average)
    Assumes input tensors are in [0, 1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred = rgb_to_grayscale(pred)
    target = rgb_to_grayscale(target)

    (_, channel, height, width) = pred.size()
    window = create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_metrics(pred, target):
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    return {'PSNR': psnr, 'SSIM': ssim}