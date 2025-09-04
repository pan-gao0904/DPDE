"""
作者： hsjxg
日期： 2025/8/5 18:17
"""
# inference.py
import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from model.UnderwaterRestorationNet import UnderwaterRestorationNet
from model.UnderwaterRestorationNetV3 import PGLANPlus
import json
import numpy as np
from tqdm import tqdm

# 从你的工具库中导入统一的指标计算函数
from utils.metrics import compute_metrics


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def save_image(tensor, path):
    tensor = tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clip
    if tensor.shape[0] == 1:  # Single-channel (grayscale)
        tensor = tensor.repeat(3, 1, 1)  # Convert to 3-channel for PIL compatibility
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(path)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def infer(args):
    config_path = "config/PGLAN.json"
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PGLANPlus().to(device)
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 创建输出子文件夹
    output_dir = args.output_dir
    t_output_dir = os.path.join(output_dir, 'T')
    a_output_dir = os.path.join(output_dir, 'A')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(t_output_dir, exist_ok=True)
    os.makedirs(a_output_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []

    image_files = [f for f in os.listdir(args.input) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Processing images"):
        file_path = os.path.join(args.input, filename)
        gt_path = os.path.join(args.gt_dir, filename)

        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Ground truth image not found for {filename}. Skipping.")
            continue

        input_tensor = load_image(file_path).to(device)
        gt_tensor = load_image(gt_path).to(device)

        with torch.no_grad():
            restored_tensor, T, A = model(input_tensor)

        # ✅ 调用统一的 compute_metrics 函数
        metrics = compute_metrics(restored_tensor, gt_tensor)

        psnr_list.append(metrics['PSNR'])
        ssim_list.append(metrics['SSIM'])

        # 保存增强图像、T 和 A 及其 R、G、B 通道
        out_path = os.path.join(output_dir, f"restored_{filename}")
        t_path = os.path.join(t_output_dir, f"T_{filename}")
        t_r_path = os.path.join(t_output_dir, f"T_r_{filename}")
        t_g_path = os.path.join(t_output_dir, f"T_g_{filename}")
        t_b_path = os.path.join(t_output_dir, f"T_b_{filename}")
        a_path = os.path.join(a_output_dir, f"A_{filename}")
        a_r_path = os.path.join(a_output_dir, f"A_r_{filename}")
        a_g_path = os.path.join(a_output_dir, f"A_g_{filename}")
        a_b_path = os.path.join(a_output_dir, f"A_b_{filename}")

        save_image(restored_tensor, out_path)
        save_image(T, t_path)
        save_image(T[:, 0:1, :, :], t_r_path)  # T R channel
        save_image(T[:, 1:2, :, :], t_g_path)  # T G channel
        save_image(T[:, 2:3, :, :], t_b_path)  # T B channel
        save_image(A, a_path)
        save_image(A[:, 0:1, :, :], a_r_path)  # A R channel
        save_image(A[:, 1:2, :, :], a_g_path)  # A G channel
        save_image(A[:, 2:3, :, :], a_b_path)  # A B channel

    if psnr_list and ssim_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print("\n--- Evaluation Results ---")
        print(f"✅ Average PSNR: {avg_psnr:.2f}")
        print(f"✅ Average SSIM: {avg_ssim:.4f}")
    else:
        print("\n❌ No images were processed for evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Underwater Image Restoration Inference')
    parser.add_argument('--input', type=str, required=True, help='Path to input folder')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--output_dir', type=str, default='outputs/test_results_PGLAN', help='Directory to save output')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth folder')
    args = parser.parse_args()

    infer(args)


    # python infer_PGLAN_T_and_A.py --input dataset/val/input --model checkpoints/best_model.pth --output_dir outputs/test_results_PGLAN --gt_dir dataset/val/gt


