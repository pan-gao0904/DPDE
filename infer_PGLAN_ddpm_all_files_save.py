import os
import time
import logging
import argparse
from PIL import Image
import torch
from torchvision import transforms
from model.UnderwaterRestorationNet import UnderwaterRestorationNet
from model.UnderwaterRestorationNetV3 import PGLANPlus

import dataset_ddpm
import model as Model
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_stage1(img_tensor, model_ckpt, device):
    config1 = load_config("config/PGLAN.json")
    model = PGLANPlus().to(device)
    checkpoint = torch.load(model_ckpt)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])  # ← 取出模型字段
    else:
        model.load_state_dict(checkpoint)  # fallback: 直接加载普通权重
    model.eval()
    with torch.no_grad():
        restored, _, _ = model(img_tensor.to(device))
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_ddpm', type=str, default='config/underwater.json', help='DDPM config file')
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/latest.pth', help='Stage1 model path')
    parser.add_argument('--output_dir', type=str, default='outputs/final_results', help='Where to save outputs')
    parser.add_argument('--phase', type=str, default='val', help='Schedule phase')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs to use (e.g., "0" or "0,1")')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_infer', action='store_true')
    args = parser.parse_args()

    args.config = args.config_ddpm  #  兼容 Logger 解析接口

    os.makedirs(args.output_dir, exist_ok=True)
    input_dir = os.path.join(args.output_dir, 'input')
    stage1_dir = os.path.join(args.output_dir, 'stage1')
    stage2_dir = os.path.join(args.output_dir, 'stage2')
    hr_dir = os.path.join(args.output_dir, 'hr')

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DDPM config
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # Setup logger
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    wandb_logger = WandbLogger(opt) if opt['enable_wandb'] else None

    # Load Dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = dataset_ddpm.create_dataset(dataset_opt, phase)
            val_loader = dataset_ddpm.create_dataloader(val_set, dataset_opt, phase)

    # Load DDPM Model
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    logger.info(' Loaded DDPM model.')

    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    for i, val_data in enumerate(val_loader):
        start_time = time.time()
        index = val_data['Index'].item()

        img_path = val_loader.dataset.sr_path[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Stage 1: Restoration
        input_tensor = ((val_data['Raw'] + 1.0) / 2.0).to(device)
        enhanced_tensor = run_stage1(input_tensor, args.stage1_ckpt, device)

        # Save stage1 result
        enhanced_img = Metrics.tensor2img(enhanced_tensor.squeeze(0).cpu(), min_max=(0, 1))
        # Metrics.save_img(enhanced_img, os.path.join(args.output_dir, f'{img_name}_stage1.png'))
        Metrics.save_img(enhanced_img, os.path.join(stage1_dir, f'{img_name}.png'))

        # Save original input
        input_img = Metrics.tensor2img(val_data['Raw'].squeeze(0))
        # Metrics.save_img(input_img, os.path.join(args.output_dir, f'{img_name}_input.png'))
        Metrics.save_img(input_img, os.path.join(input_dir, f'{img_name}.png'))

        # Save reference GT
        hr_img = Metrics.tensor2img(val_data['Ref'].squeeze(0))
        # Metrics.save_img(hr_img, os.path.join(args.output_dir, f'{img_name}_hr.png'))
        Metrics.save_img(hr_img, os.path.join(hr_dir, f'{img_name}.png'))

        # Stage 2: DDPM
        val_data['Raw'] = enhanced_tensor * 2.0 - 1.0
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)

        visuals = diffusion.get_current_visuals(need_LR=False)
        final_img = Metrics.tensor2img(visuals['Enhanced'][-1])

        # Metrics.save_img(final_img, os.path.join(args.output_dir, f'{img_name}_stage2.png'))
        Metrics.save_img(final_img, os.path.join(stage2_dir, f'{img_name}.png'))

        end_time = time.time()
        print(f"🕒 {img_name} 处理耗时: {end_time - start_time:.2f} 秒")

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(enhanced_img, final_img, input_img)

    print(f"\n 所有图像处理完成，结果已保存至：{args.output_dir}")


if __name__ == '__main__':
    main()


# python infer_PGLAN_ddpm_all_files_save.py --config_ddpm config/underwater.json --stage1_ckpt checkpoints/best_model.pth --output_dir outputs/final_results
