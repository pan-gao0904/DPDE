# 代码内容：
"""
作者： hsjxg
日期： 2025/8/5 17:18
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Shared Encoder
# -------------------------
class SharedEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encoder(x)


# -------------------------
# Transmission Decoder
# -------------------------
class TDecoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()  # T in range (0,1)
        )

    def forward(self, feat):
        return self.decoder(feat)


# -------------------------
# Background Light Decoder
# -------------------------
class ADecoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()  # A in range (0,1)
        )

    def forward(self, feat):
        return self.decoder(feat)


# -------------------------
# Physics-based Reconstruction
# -------------------------
def physics_reconstruction(I, T, A, epsilon=1e-2):
    J = (I - A) / torch.clamp(T, min=epsilon) + A
    J = torch.clamp(J, 0, 1)  # optional
    return J


# -------------------------
# Full Model
# -------------------------
class UnderwaterRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.T_net = TDecoder()
        self.A_net = ADecoder()

    def forward(self, I):
        feat = self.encoder(I)
        T = self.T_net(feat)  # shape [B,3,H,W]
        A = self.A_net(feat)  # shape [B,3,H,W]
        J_hat = physics_reconstruction(I, T, A)
        return J_hat, T, A


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    model = UnderwaterRestorationNet()
    dummy_input = torch.randn(1, 3, 256, 256)  # [B,C,H,W]
    J_hat, T, A = model(dummy_input)
    print("Output image shape:", J_hat.shape)
    print("T map shape:", T.shape)
    print("A map shape:", A.shape)
