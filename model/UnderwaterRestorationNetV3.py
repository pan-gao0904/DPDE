import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Multi-Scale Attention Block with SE Attention ==========
class MSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = out_channels // reduction

        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv3x3_pw = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv5x5_pw = nn.Conv2d(in_channels, mid_channels, 1)

        self.fuse = nn.Conv2d(mid_channels * 3, out_channels, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        # ⬇️ Projection to match dimensions if needed
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1x1(x)
        c3 = self.conv3x3_pw(self.conv3x3(x))
        c5 = self.conv5x5_pw(self.conv5x5(x))
        fused = torch.cat([c1, c3, c5], dim=1)

        out = self.fuse(fused)
        attn = self.se(out)
        out = out * attn

        return self.relu(out + self.shortcut(x))  # ✅ 通道对齐后可加法



# ========== Residual Block ==========
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


# ========== Encoder ==========
class SharedEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.msa1 = MSABlock(base_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.msa2 = MSABlock(base_channels*2, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.msa3 = MSABlock(base_channels*4, base_channels*4)

    def forward(self, x):
        feat1 = self.msa1(self.conv1(x))
        feat2 = self.msa2(self.conv2(self.pool1(feat1)))
        feat3 = self.msa3(self.conv3(self.pool2(feat2)))
        return feat1, feat2, feat3


# ========== T Decoder ==========
class TDecoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.msa1 = MSABlock(base_channels*4, base_channels*2)

        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.msa2 = MSABlock(base_channels*2, base_channels)

        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f1, f2, f3):
        x = self.up1(f3)
        x = self.msa1(torch.cat([x, f2], dim=1))
        x = self.up2(x)
        x = self.msa2(torch.cat([x, f1], dim=1))
        t = self.out(x)
        return 0.8 * t + 0.1  # 保证范围 [0.1, 0.9]


# ========== A Decoder ==========
class ADecoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(base_channels*4, base_channels, 4, stride=4)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f3):
        a = self.up(f3)
        return 0.6 * self.out(a) + 0.2  # 保证范围 [0.2, 0.8]


# ========== Physics Module ==========
class PhysicsReconstruction(nn.Module):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, I, T, A):
        J = (I - A) / torch.clamp(T, min=self.epsilon) + A
        return torch.clamp(J, 0, 1)


# ========== Refinement ==========
class Refinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            ResidualBlock(3),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.refine(x)


# ========== Full Model ==========
class PGLANPlus(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.encoder = SharedEncoder(base_channels=base_channels)
        self.T_decoder = TDecoder(base_channels=base_channels)
        self.A_decoder = ADecoder(base_channels=base_channels)
        self.physics = PhysicsReconstruction()
        self.refine = Refinement()

    def forward(self, I):
        f1, f2, f3 = self.encoder(I)
        T = self.T_decoder(f1, f2, f3)
        A = self.A_decoder(f3)
        J = self.physics(I, T, A)
        J = self.refine(J)
        return J, T, A


if __name__ == "__main__":
    model = PGLANPlus()
    dummy_input = torch.randn(1, 3, 256, 256)
    J_hat, T, A = model(dummy_input)
    print("Output image shape:", J_hat.shape)
    print("T map shape:", T.shape)
    print("A map shape:", A.shape)
