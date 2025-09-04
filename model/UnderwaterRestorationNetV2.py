# 代码内容：
"""
作者： hsjxg
日期： 2025/8/6 10:03
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightweight Multi-Scale Attention Block
class MSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = out_channels // reduction

        # Depthwise + Pointwise Convs
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, 1)

        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv3x3_pw = nn.Conv2d(in_channels, mid_channels, 1)

        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv5x5_pw = nn.Conv2d(in_channels, mid_channels, 1)

        # Attention network
        self.attention = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(mid_channels * 3, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = self.conv1x1(x)
        c3 = self.conv3x3_pw(self.conv3x3(x))
        c5 = self.conv5x5_pw(self.conv5x5(x))
        fused = torch.cat([c1, c3, c5], dim=1)  # shape: [B, 3*mid_channels]

        attn = self.attention(fused)  # shape: [B, out_channels]
        proj = self.out_conv(fused)   # shape: [B, out_channels]

        out = proj * attn             # element-wise multiplication
        out = self.bn(out)
        out = self.relu(out)
        return out + x  # residual



# Shared Encoder with Lightweight Convolutions
class SharedEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, base_channels, 1),  # Pointwise
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.msa1 = MSABlock(base_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels),
            nn.Conv2d(base_channels, base_channels*2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.msa2 = MSABlock(base_channels*2, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat1 = self.msa1(feat1)
        feat2 = self.pool1(feat1)
        feat3 = self.conv2(feat2)
        feat3 = self.msa2(feat3)
        return feat1, feat3


# Transmission Decoder
class TDecoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.msa = MSABlock(base_channels * 2, base_channels * 2)
        self.reduce = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),  # ← 修改为输出3通道
            nn.Sigmoid()
        )

    def forward(self, feat_low, feat_high):
        up = self.up1(feat_high)
        fused = torch.cat([up, feat_low], dim=1)
        fused = self.msa(fused)
        fused = self.reduce(fused)
        return self.out(fused)


# Background Light Decoder
class ADecoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),  # 输出为3通道图像大小
            nn.Sigmoid()
        )

    def forward(self, feat_high):
        x = self.up1(feat_high)  # [B, base_channels, H, W]
        return self.out(x)       # [B, 3, H, W]


# Physics-Guided Reconstruction Module
class PhysicsReconstruction(nn.Module):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, I, T, A):
        # 现在 A 已经是 [B, 3, H, W]，无需 expand
        J = (I - A) / torch.clamp(T, min=self.epsilon) + A
        return torch.clamp(J, 0, 1)

# PGLAN: Physically-Guided Lightweight Attention Network
class PGLAN(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.encoder = SharedEncoder(base_channels=base_channels)
        self.T_decoder = TDecoder(base_channels=base_channels)
        self.A_decoder = ADecoder(base_channels=base_channels)
        self.physics = PhysicsReconstruction()

    def forward(self, I):
        feat_low, feat_high = self.encoder(I)
        T = self.T_decoder(feat_low, feat_high)
        A = self.A_decoder(feat_high)
        J = self.physics(I, T, A)
        return J, T, A


# Example Usage
if __name__ == "__main__":
    model = PGLAN(base_channels=32)
    dummy_input = torch.randn(1, 3, 256, 256)
    J_hat, T, A = model(dummy_input)
    print("Output image shape:", J_hat.shape)
    print("T map shape:", T.shape)
    print("A map shape:", A.shape)
