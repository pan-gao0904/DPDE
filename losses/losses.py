import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


# --------- VGG感知损失（多层） ---------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        # 加载预训练的VGG-16模型
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.blocks = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16],  # relu3_3
            vgg[16:23],  # relu4_3
        ])
        # 对不同VGG层输出的特征图赋予不同的权重
        self.weights = [1.0, 1.0, 1.0, 0.5]  # relu4_3 权重较小
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        self.blocks.to(device)

        # VGG所需的归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        loss = 0
        for block, w in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            # 计算L1损失
            loss += w * F.l1_loss(x, y)
        return loss


# --------- 综合损失函数（集成感知损失和多尺度L1） ---------
class CombinedLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss(device=device)

    def forward(self, input_I, pred_J, pred_T, pred_A, target_J):
        """
        集成了多种损失项的综合损失函数。
        """
        # 重建损失（L1和L2）
        l1_recon_loss = self.l1(pred_J, target_J)
        l2_recon_loss = self.mse(pred_J, target_J)

        # 感知损失
        perceptual_loss = self.perceptual_loss(pred_J, target_J)

        # 多尺度L1损失
        multiscale_l1_loss = self._multiscale_l1_loss(pred_J, target_J)

        # SSIM损失
        ssim_loss_val = self._ssim_simple(pred_J, target_J)
        ssim_loss = 1 - ssim_loss_val

        # 物理约束损失
        t_constraint = F.relu(0.01 - pred_T).mean() + F.relu(pred_T - 0.99).mean()
        a_constraint = F.relu(0.01 - pred_A).mean() + F.relu(pred_A - 0.99).mean()

        # --------------------- 损失函数权重配置 ---------------------
        # 这是我的建议权重配置，你可以根据实验效果进行调整
        lambda_l1 = 5
        lambda_perceptual = 0.1
        lambda_multiscale = 0.1
        lambda_ssim = 0.5
        lambda_physics = 0.001

        # 总损失 = 重建损失 + 感知损失 + 多尺度L1 + SSIM + 物理约束
        total_loss = (
                lambda_l1 * l1_recon_loss +
                lambda_perceptual * perceptual_loss +
                lambda_multiscale * multiscale_l1_loss +
                lambda_ssim * ssim_loss +
                lambda_physics * (t_constraint + a_constraint)
        )
        # -----------------------------------------------------------

        return total_loss, {
            'l1_recon': l1_recon_loss.item(),
            'l2_recon': l2_recon_loss.item(),
            'perceptual': perceptual_loss.item(),
            'multiscale_l1': multiscale_l1_loss.item(),
            'ssim_val': ssim_loss_val.item(),  # 返回SSIM值方便观察
            't_constraint': t_constraint.item(),
            'a_constraint': a_constraint.item(),
            'total': total_loss.item()
        }

    def _ssim_simple(self, img1, img2):
        """简化SSIM计算"""
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def _multiscale_l1_loss(self, pred, target):
        """多尺度L1损失（提升不同尺度的细节）"""
        loss = 0
        for scale in [1, 0.5, 0.25]:
            if scale != 1:
                h, w = int(pred.size(2) * scale), int(pred.size(3) * scale)
                pred_scaled = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
            else:
                pred_scaled, target_scaled = pred, target
            loss += F.l1_loss(pred_scaled, target_scaled)
        return loss / 3