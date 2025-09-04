import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG Perceptual Loss (Using deeper layers for texture enhancement)
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()  # Use conv4_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        return F.l1_loss(self.vgg(x_norm), self.vgg(y_norm))

# Edge Loss for sharpness
def edge_loss(pred, gt, weight=0.1):
    pred_grad_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    pred_grad_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    gt_grad_x = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    gt_grad_y = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    return weight * (F.l1_loss(pred_grad_x, gt_grad_x) + F.l1_loss(pred_grad_y, gt_grad_y))

# Combined Loss Function
def get_total_loss(J_pred, J_gt, T=None, A=None, input_img=None, perceptual=None,
                   epoch=0, total_epochs=200, lambda_l1=1.0, lambda_perceptual_base=0.7,
                   lambda_phys_base=0.5, lambda_edge=0.1):
    # Pixel-wise L1 loss
    loss = lambda_l1 * F.l1_loss(J_pred, J_gt)

    # Dynamic weights
    lambda_phys = lambda_phys_base * (1 - epoch / total_epochs) + 0.2
    lambda_perceptual = lambda_perceptual_base * (epoch / total_epochs) + 0.7

    # Perceptual loss for texture and sharpness
    if perceptual is not None and lambda_perceptual > 0:
        loss += lambda_perceptual * perceptual(J_pred, J_gt)

    # Physical reconstruction loss
    if input_img is not None and T is not None and A is not None:
        I_reconstructed = J_pred * T + A * (1 - T)
        loss += lambda_phys * F.l1_loss(I_reconstructed, input_img)

    # Edge loss for enhanced sharpness
    loss += edge_loss(J_pred, J_gt, lambda_edge)

    return loss

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual = VGGPerceptualLoss(device=device)
    J_pred = torch.randn(1, 3, 256, 256).to(device)
    J_gt = torch.randn(1, 3, 256, 256).to(device)
    T = torch.randn(1, 3, 256, 256).to(device)
    A = torch.randn(1, 3, 256, 256).to(device)
    input_img = torch.randn(1, 3, 256, 256).to(device)
    loss = get_total_loss(J_pred, J_gt, T, A, input_img, perceptual, epoch=1, total_epochs=200)
    print("Total Loss:", loss.item())