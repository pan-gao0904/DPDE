import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from skimage.color import rgb2lab
import model.networks as networks
from .base_model import BaseModel
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
import kornia

logger = logging.getLogger('base')

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        self.layers = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:16], # conv3_3
            vgg[16:23] # conv4_3
        ]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss(reduction='mean').to(device)

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)  # Convert grayscale to RGB if needed
            target = target.repeat(1, 3, 1, 1)
        input = (input + 1) / 2.0  # Normalize to [0, 1]
        target = (target + 1) / 2.0
        loss = 0.0
        for layer in self.layers:
            input = layer(input)
            target = layer(target)
            loss += self.mse_loss(input, target)
        return loss

def ssim_loss(img1, img2, window_size=11, size_average=True):
    """
    Custom SSIM loss implementation to avoid padding issue.
    Args:
        img1: Tensor, first image (e.g., x_start)
        img2: Tensor, second image (e.g., x_0_pred)
        window_size: int, Gaussian window size (default: 11)
        size_average: bool, whether to average the loss
    Returns:
        SSIM score (1 - SSIM for loss)
    """
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g[:, None] * g[None, :]  # 2D Gaussian

    channel = img1.shape[1]
    window = gaussian_window(window_size, 1.5).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)

    padding = window_size // 2  # Integer padding
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return 1 - ssim_map.mean()  # Return 1 - SSIM as loss
    return 1 - ssim_map

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # Define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # Set loss functions
        self.set_loss()
        self.l1_loss = nn.L1Loss(reduction='mean').to(self.device)
        self.perceptual_loss = VGGPerceptualLoss(self.device)
        self.ssim_loss = ssim_loss  # Use custom SSIM loss
        self.color_loss = nn.MSELoss(reduction='mean').to(self.device)  # MSE for Lab color loss

        # Loss weights from opt
        self.l1_weight = opt['train'].get('l1_weight', 1.0)
        self.perceptual_weight = opt['train'].get('perceptual_weight', 0.1)
        self.ssim_weight = opt['train'].get('ssim_weight', 0.1)
        self.color_weight = opt['train'].get('color_weight', 0.1)
        self.stage = opt['train'].get('stage', 1)  # Default to stage 1
        logger.info(f'Loss weights: l1={self.l1_weight}, perceptual={self.perceptual_weight}, '
                    f'ssim={self.ssim_weight}, color={self.color_weight}, stage={self.stage}')

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            # ✅ 添加余弦退火调度器（从配置中读取）
            scheduler_cfg = opt['train'].get('scheduler', {})
            if scheduler_cfg.get('type') == 'cosine':
                T_max = scheduler_cfg.get('T_max', opt['train']['n_iter'])  # fallback to total iters
                eta_min = scheduler_cfg.get('eta_min', 1e-6)
                self.scheduler = CosineAnnealingLR(self.optG, T_max=T_max, eta_min=eta_min)
                logger.info(f'Using CosineAnnealingLR scheduler: T_max={T_max}, eta_min={eta_min}')
            else:
                self.scheduler = None
            self.log_dict = OrderedDict()
        self.load_network()

        # self.print_network()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, flag=None):
        if flag is None:
            self.optG.zero_grad()
            # Get input dataset_ddpm
            x_start = self.data['Ref']  # Ground-truth HR image
            condition_x = self.data['Raw']  # Condition (low-res or dehazed) image
            b, c, h, w = x_start.shape
            num_timesteps = self.netG.module.num_timesteps if isinstance(self.netG, nn.DataParallel) else self.netG.num_timesteps
            t = torch.randint(0, num_timesteps, (b,), device=x_start.device).long()

            # Generate noisy image x_t
            noise = torch.randn_like(x_start)
            if isinstance(self.netG, nn.DataParallel):
                x_noisy = self.netG.module.q_sample(x_start=x_start, t=t, noise=noise)
            else:
                x_noisy = self.netG.q_sample(x_start=x_start, t=t, noise=noise)

            # Predict noise x_recon
            if isinstance(self.netG, nn.DataParallel):
                x_recon = self.netG.module.denoise_fn(
                    torch.cat([condition_x, x_noisy], dim=1), t)
            else:
                x_recon = self.netG.denoise_fn(
                    torch.cat([condition_x, x_noisy], dim=1), t)

            # Compute diffusion loss (L1 loss between noise and predicted noise)
            l_diffusion = self.l1_loss(noise, x_recon)

            # Predict x_0 using predict_start_from_noise
            if isinstance(self.netG, nn.DataParallel):
                x_0_pred = self.netG.module.predict_start_from_noise(x_noisy, t, x_recon)
            else:
                x_0_pred = self.netG.predict_start_from_noise(x_noisy, t, x_recon)
            x_0_pred = x_0_pred.clamp(-1., 1.)  # Ensure pixel range consistency

            # Compute additional losses
            l_perceptual = self.perceptual_loss(x_0_pred, x_start)
            l_ssim = self.ssim_loss(x_start, x_0_pred, window_size=11, size_average=True)

            # Combine losses based on stage
            if self.stage == 1:
                total_loss = (self.l1_weight * l_diffusion +
                              self.perceptual_weight * l_perceptual +
                              self.ssim_weight * l_ssim)
                l_color = torch.tensor(0.0, device=self.device)
            else:  # Stage 2
                # Convert to Lab color space (input range [0, 1])
                x = (x_0_pred + 1) / 2.0  # Normalize to [0, 1]
                x=x.clamp(0.0,1.0).float()
                # x_0_pred_lab = torch.tensor(rgb2lab(x_0_pred_cpu.cpu()), dtype=torch.float, device=self.device)
                x_0_pred_lab = kornia.color.rgb_to_lab(x)
                x_start_lab = torch.tensor(rgb2lab((x_start + 1) / 2.0), dtype=torch.float, device=self.device)
                l_color = self.color_loss(x_0_pred_lab[:, 1:3], x_start_lab[:, 1:3])
                total_loss = (self.l1_weight * l_diffusion +
                              self.perceptual_weight * l_perceptual +
                              self.ssim_weight * l_ssim +
                              self.color_weight * l_color)

            total_loss.backward()
            self.optG.step()
            self.scheduler.step()
            # Log losses
            self.log_dict['l_diffusion'] = l_diffusion.item()
            self.log_dict['l_perceptual'] = l_perceptual.item()
            self.log_dict['l_ssim'] = l_ssim.item()
            self.log_dict['l_color'] = l_color.item()
            self.log_dict['total_loss'] = total_loss.item()

    def optimize_parameters2(self):
        self.optG.zero_grad()
        # Get input dataset_ddpm
        x_start = self.data['Ref']
        condition_x = self.data['Raw']
        b, c, h, w = x_start.shape
        num_timesteps = self.netG.module.num_timesteps if isinstance(self.netG, nn.DataParallel) else self.netG.num_timesteps
        t = torch.randint(0, num_timesteps, (b,), device=x_start.device).long()

        # Generate noisy image x_t
        noise = torch.randn_like(x_start)
        if isinstance(self.netG, nn.DataParallel):
            x_noisy = self.netG.module.q_sample(x_start=x_start, t=t, noise=noise)
        else:
            x_noisy = self.netG.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise x_recon
        if isinstance(self.netG, nn.DataParallel):
            x_recon = self.netG.module.denoise_fn(
                torch.cat([condition_x, x_noisy], dim=1), t)
        else:
            x_recon = self.netG.denoise_fn(
                torch.cat([condition_x, x_noisy], dim=1), t)

        # Compute diffusion loss (L1 loss between noise and predicted noise)
        l_diffusion = self.l1_loss(noise, x_recon)
        l_diffusion = l_diffusion.sum() / int(b * c * h * w)  # Normalize

        # Predict x_0 using predict_start_from_noise
        if isinstance(self.netG, nn.DataParallel):
            x_0_pred = self.netG.module.predict_start_from_noise(x_noisy, t, x_recon)
        else:
            x_0_pred = self.netG.predict_start_from_noise(x_noisy, t, x_recon)
        x_0_pred = x_0_pred.clamp(-1., 1.)

        # Compute additional losses
        l_perceptual = self.perceptual_loss(x_0_pred, x_start)
        l_ssim = self.ssim_loss(x_start, x_0_pred, window_size=11, size_average=True)

        if self.stage == 1:
            total_loss = (self.l1_weight * l_diffusion +
                          self.perceptual_weight * l_perceptual +
                          self.ssim_weight * l_ssim)
            l_color = torch.tensor(0.0, device=self.device)
        else:  # Stage 2
            x_0_pred_lab = torch.tensor(rgb2lab((x_0_pred + 1) / 2.0), dtype=torch.float, device=self.device)
            x_start_lab = torch.tensor(rgb2lab((x_start + 1) / 2.0), dtype=torch.float, device=self.device)
            l_color = self.color_loss(x_0_pred_lab[:, 1:3], x_start_lab[:, 1:3])
            total_loss = (self.l1_weight * l_diffusion +
                          self.perceptual_weight * l_perceptual +
                          self.ssim_weight * l_ssim +
                          self.color_weight * l_color)

        total_loss.backward()
        self.optG.step()
        self.scheduler.step()
        # Log losses
        self.log_dict['l_diffusion'] = l_diffusion.item()
        self.log_dict['l_perceptual'] = l_perceptual.item()
        self.log_dict['l_ssim'] = l_ssim.item()
        self.log_dict['l_color'] = l_color.item()
        self.log_dict['total_loss'] = total_loss.item()

    def test(self, cand=None, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(self.data, continous)
            else:
                self.SR = self.netG.super_resolution(self.data, continous, cand=cand)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['Enhanced'] = self.SR.detach().float().cpu()

            out_dict['Ref'] = self.data['Ref'].detach().float().cpu()
            out_dict['Raw'] = self.data['Raw'].detach().float().cpu()
            # if need_LR and 'LR' in self.data:
            #     out_dict['LR'] = self.data['LR'].detach().float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path, weights_only=True), strict=True)
            if self.opt['phase'] == 'train' and os.path.exists(opt_path):
                logger.info('Found optimizer state, loading...')
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
            else:
                logger.warning('No optimizer state found at {}. Starting fine-tuning from scratch.'.format(opt_path))
                self.begin_step = 0
                self.begin_epoch = 0

#
# import logging
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# from skimage.color import rgb2lab
# import model.networks as networks
# from .base_model import BaseModel
# import kornia.color as kcolor
# from skimage.metrics import structural_similarity as sk_ssim
# import numpy as np
#
#
# logger = logging.getLogger('base')
#
# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, device):
#         super(VGGPerceptualLoss, self).__init__()
#         vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features.eval().to(device)
#         self.layers = nn.ModuleList([
#             vgg[:4],   # conv1_2
#             vgg[4:9],  # conv2_2
#             vgg[9:16], # conv3_3
#             vgg[16:23] # conv4_3
#         ]).eval()
#         for param in self.layers.parameters():
#             param.requires_grad = False
#         self.mse_loss = nn.MSELoss(reduction='mean').to(device)
#
#     def forward(self, input, target):
#         if input.shape[1] != 3:
#             input = input.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#         input = (input + 1) / 2.0  # Normalize to [0, 1]
#         target = (target + 1) / 2.0
#         loss = 0.0
#         for layer in self.layers:
#             input = layer(input)
#             target = layer(target)
#             loss += self.mse_loss(input, target)
#         return loss
#
# def ms_ssim_loss(img1, img2, window_size=11, size_average=True):
#     """Multi-Scale SSIM loss."""
#     weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
#     levels = len(weights)
#     mssim = []
#     for _ in range(levels):
#         ssim_val = 1 - ssim_loss(img1, img2, window_size, size_average)
#         mssim.append(ssim_val)
#         img1 = F.avg_pool2d(img1, 2)
#         img2 = F.avg_pool2d(img2, 2)
#     mssim = torch.stack(mssim)
#     weights = torch.tensor(weights, device=img1.device)
#     return 1 - (mssim * weights).sum()
#
# def ssim_loss(img1, img2, window_size=11, size_average=True):
#     def gaussian_window(size, sigma):
#         coords = torch.arange(size, dtype=torch.float32, device=img1.device)
#         coords -= size // 2
#         g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#         g /= g.sum()
#         return g[:, None] * g[None, :]
#     channel = img1.shape[1]
#     window = gaussian_window(window_size, 1.5)
#     window = window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
#     padding = window_size // 2
#     mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean() if size_average else ssim_map
#
# def charbonnier_loss(pred, target, eps=1e-6):
#     """Charbonnier loss for robust pixel supervision."""
#     return torch.mean(torch.sqrt((pred - target) ** 2 + eps))
#
# def laplacian_loss(pred, target, weight=1.0):
#     """Laplacian loss for high-frequency consistency."""
#     kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=pred.device)
#     kernel = kernel.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
#     pred_lap = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
#     target_lap = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
#     return weight * F.l1_loss(pred_lap, target_lap)
#
# def color_distance(img1, img2):
#     """Compute color distance in Lab space (a,b channels)."""
#     img1_lab = kcolor.rgb_to_lab((img1 + 1) / 2.0)
#     img2_lab = kcolor.rgb_to_lab((img2 + 1) / 2.0)
#     ab1 = img1_lab[:, 1:3].mean(dim=[2, 3])
#     ab2 = img2_lab[:, 1:3].mean(dim=[2, 3])
#     return torch.norm(ab1 - ab2, dim=1).mean()
#
# class DDPM(BaseModel):
#     def __init__(self, opt):
#         super(DDPM, self).__init__(opt)
#         self.netG = self.set_device(networks.define_G(opt))
#         self.schedule_phase = None
#         self.set_loss()
#         self.l1_loss = nn.L1Loss(reduction='mean').to(self.device)
#         self.perceptual_loss = VGGPerceptualLoss(self.device)
#         self.ms_ssim_loss = ms_ssim_loss
#         self.charbonnier_loss = charbonnier_loss
#         self.laplacian_loss = laplacian_loss
#         self.color_distance = color_distance
#         self.l1_weight = opt['train'].get('l1_weight', 1.0)
#         self.perceptual_weight = opt['train'].get('perceptual_weight', 0.1)
#         self.ms_ssim_weight = opt['train'].get('ms_ssim_weight', 0.1)
#         self.laplacian_weight = opt['train'].get('laplacian_weight', 0.5)
#         self.color_weight = opt['train'].get('color_weight', 0.3)
#         self.identity_weight_base = opt['train'].get('identity_weight_base', 0.5)
#         self.color_threshold = opt['train'].get('color_threshold', 5.0)
#         self.stage = opt['train'].get('stage', 2)  # Stage 2 for color/style transfer
#         logger.info(f'Loss weights: l1={self.l1_weight}, perceptual={self.perceptual_weight}, '
#                     f'ms_ssim={self.ms_ssim_weight}, laplacian={self.laplacian_weight}, '
#                     f'color={self.color_weight}, identity_base={self.identity_weight_base}, '
#                     f'color_threshold={self.color_threshold}, stage={self.stage}')
#         self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
#         if self.opt['phase'] == 'train':
#             self.netG.train()
#             if opt['model']['finetune_norm']:
#                 optim_params = []
#                 for k, v in self.netG.named_parameters():
#                     v.requires_grad = False
#                     if k.find('transformer') >= 0:
#                         v.requires_grad = True
#                         v.data.zero_()
#                         optim_params.append(v)
#                         logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
#             else:
#                 optim_params = list(self.netG.parameters())
#             self.optG = torch.optim.Adam(optim_params, lr=opt['train']['optimizer']['lr'])
#             self.log_dict = OrderedDict()
#         self.load_network()
#
#     def set_requires_grad(self, nets, requires_grad=False):
#         if not isinstance(nets, list):
#             nets = [nets]
#         for net in nets:
#             if net is not None:
#                 for param in net.parameters():
#                     param.requires_grad = requires_grad
#
#     def feed_data(self, data):
#         self.data = self.set_device(data)
#
#     def optimize_parameters(self, flag=None):
#         if flag is None:
#             self.optG.zero_grad()
#             x_start = self.data['Ref']  # Reference image
#             condition_x = self.data['Raw']  # Go-scattered image
#             b, c, h, w = x_start.shape
#             num_timesteps = self.netG.module.num_timesteps if isinstance(self.netG, nn.DataParallel) else self.netG.num_timesteps
#             t = torch.randint(0, num_timesteps, (b,), device=x_start.device).long()
#             noise = torch.randn_like(x_start)
#             if isinstance(self.netG, nn.DataParallel):
#                 x_noisy = self.netG.module.q_sample(x_start=x_start, t=t, noise=noise)
#                 x_recon = self.netG.module.denoise_fn(torch.cat([condition_x, x_noisy], dim=1), t)
#                 x_0_pred = self.netG.module.predict_start_from_noise(x_noisy, t, x_recon)
#             else:
#                 x_noisy = self.netG.q_sample(x_start=x_start, t=t, noise=noise)
#                 x_recon = self.netG.denoise_fn(torch.cat([condition_x, x_noisy], dim=1), t)
#                 x_0_pred = self.netG.predict_start_from_noise(x_noisy, t, x_recon)
#             x_0_pred = x_0_pred.clamp(-1., 1.)
#             l_diffusion = self.l1_loss(noise, x_recon)
#             x_0_pred_lab = kcolor.rgb_to_lab((x_0_pred + 1) / 2.0)
#             x_start_lab = kcolor.rgb_to_lab((x_start + 1) / 2.0)
#             condition_x_lab = kcolor.rgb_to_lab((condition_x + 1) / 2.0)
#             l_l1 = self.l1_loss(x_0_pred_lab[:, 0:1], condition_x_lab[:, 0:1])
#             l_ms_ssim = self.ms_ssim_loss(x_0_pred_lab[:, 0:1], condition_x_lab[:, 0:1])
#             l_laplacian = self.laplacian_loss(x_0_pred_lab[:, 0:1], condition_x_lab[:, 0:1], self.laplacian_weight)
#             color_residual = (x_0_pred_lab[:, 1:3] - condition_x_lab[:, 1:3]) - (x_start_lab[:, 1:3] - condition_x_lab[:, 1:3])
#             l_color_residual = self.l1_loss(color_residual, torch.zeros_like(color_residual))
#             l_perceptual_ab = self.perceptual_loss(x_0_pred, x_start)
#             dist = self.color_distance(condition_x, x_start)
#             identity_weight = self.identity_weight_base if dist < self.color_threshold else 0.0
#             l_identity = identity_weight * self.charbonnier_loss(x_0_pred, condition_x)
#             total_loss = (
#                 self.l1_weight * l_diffusion +
#                 self.l1_weight * l_l1 +
#                 self.ms_ssim_weight * l_ms_ssim +
#                 l_laplacian +
#                 self.color_weight * l_color_residual +
#                 self.perceptual_weight * l_perceptual_ab +
#                 l_identity
#             )
#             total_loss.backward()
#             self.optG.step()
#             self.log_dict['l_diffusion'] = l_diffusion.item()
#             self.log_dict['l_l1'] = l_l1.item()
#             self.log_dict['l_ms_ssim'] = l_ms_ssim.item()
#             self.log_dict['l_laplacian'] = l_laplacian.item()
#             self.log_dict['l_color_residual'] = l_color_residual.item()
#             self.log_dict['l_perceptual_ab'] = l_perceptual_ab.item()
#             self.log_dict['l_identity'] = l_identity.item()
#             self.log_dict['total_loss'] = total_loss.item()
#
#     def optimize_parameters2(self):
#         self.optG.zero_grad()
#         x_start = self.data['Ref']
#         condition_x = self.data['Raw']
#         b, c, h, w = x_start.shape
#         num_timesteps = self.netG.module.num_timesteps if isinstance(self.netG, nn.DataParallel) else self.netG.num_timesteps
#         t = torch.randint(0, num_timesteps, (b,), device=x_start.device).long()
#         noise = torch.randn_like(x_start)
#         if isinstance(self.netG, nn.DataParallel):
#             x_noisy = self.netG.module.q_sample(x_start=x_start, t=t, noise=noise)
#             x_recon = self.netG.module.denoise_fn(torch.cat([condition_x, x_noisy], dim=1), t)
#             x_0_pred = self.netG.module.predict_start_from_noise(x_noisy, t, x_recon)
#         else:
#             x_noisy = self.netG.q_sample(x_start=x_start, t=t, noise=noise)
#             x_recon = self.netG.denoise_fn(torch.cat([condition_x, x_noisy], dim=1), t)
#             x_0_pred = self.netG.predict_start_from_noise(x_noisy, t, x_recon)
#         x_0_pred = x_0_pred.clamp(-1., 1.)
#         l_diffusion = self.l1_loss(noise, x_recon).sum() / int(b * c * h * w)
#         x_0_pred_lab = torch.tensor(rgb2lab((x_0_pred + 1) / 2.0), dtype=torch.float, device=self.device)
#         x_start_lab = torch.tensor(rgb2lab((x_start + 1) / 2.0), dtype=torch.float, device=self.device)
#         condition_x_lab = torch.tensor(rgb2lab((condition_x + 1) / 2.0), dtype=torch.float, device=self.device)
#         l_l1 = self.l1_loss(x_0_pred_lab[:, 0:1], condition_x_lab[:, 0:1])
#         l_ms_ssim = self.ms_ssim_loss(x_0_pred_lab[:, 0:1], condition_x_lab[:, 0:1])
#         l_laplacian = self.laplacian_loss(x_0_pred_lab[:, 0:1], condition_x_lab[:, 0:1], self.laplacian_weight)
#         color_residual = (x_0_pred_lab[:, 1:3] - condition_x_lab[:, 1:3]) - (x_start_lab[:, 1:3] - condition_x_lab[:, 1:3])
#         l_color_residual = self.l1_loss(color_residual, torch.zeros_like(color_residual))
#         l_perceptual_ab = self.perceptual_loss(x_0_pred, x_start)
#         dist = self.color_distance(condition_x, x_start)
#         identity_weight = self.identity_weight_base if dist < self.color_threshold else 0.0
#         l_identity = identity_weight * self.charbonnier_loss(x_0_pred, condition_x)
#         total_loss = (
#             self.l1_weight * l_diffusion +
#             self.l1_weight * l_l1 +
#             self.ms_ssim_weight * l_ms_ssim +
#             l_laplacian +
#             self.color_weight * l_color_residual +
#             self.perceptual_weight * l_perceptual_ab +
#             l_identity
#         )
#         total_loss.backward()
#         self.optG.step()
#         self.log_dict['l_diffusion'] = l_diffusion.item()
#         self.log_dict['l_l1'] = l_l1.item()
#         self.log_dict['l_ms_ssim'] = l_ms_ssim.item()
#         self.log_dict['l_laplacian'] = l_laplacian.item()
#         self.log_dict['l_color_residual'] = l_color_residual.item()
#         self.log_dict['l_perceptual_ab'] = l_perceptual_ab.item()
#         self.log_dict['l_identity'] = l_identity.item()
#         self.log_dict['total_loss'] = total_loss.item()
#
#     def test(self, cand=None, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.super_resolution(self.data, continous)
#             else:
#                 self.SR = self.netG.super_resolution(self.data, continous, cand=cand)
#         self.netG.train()
#
#     def sample(self, batch_size=1, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.sample(batch_size, continous)
#             else:
#                 self.SR = self.netG.sample(batch_size, continous)
#         self.netG.train()
#
#     def set_loss(self):
#         if isinstance(self.netG, nn.DataParallel):
#             self.netG.module.set_loss(self.device)
#         else:
#             self.netG.set_loss(self.device)
#
#     def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
#         if self.schedule_phase is None or self.schedule_phase != schedule_phase:
#             self.schedule_phase = schedule_phase
#             if isinstance(self.netG, nn.DataParallel):
#                 self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
#             else:
#                 self.netG.set_new_noise_schedule(schedule_opt, self.device)
#
#     def get_current_log(self):
#         return self.log_dict
#
#     def get_current_visuals(self, need_LR=True, sample=False):
#         out_dict = OrderedDict()
#         if sample:
#             out_dict['SAM'] = self.SR.detach().float().cpu()
#         else:
#             out_dict['Enhanced'] = self.SR.detach().float().cpu()
#             out_dict['Ref'] = self.data['Ref'].detach().float().cpu()
#             out_dict['Raw'] = self.data['Raw'].detach().float().cpu()
#         return out_dict
#
#     def print_network(self):
#         s, n = self.get_network_description(self.netG)
#         if isinstance(self.netG, nn.DataParallel):
#             net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
#         else:
#             net_struc_str = '{}'.format(self.netG.__class__.__name__)
#         logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
#         logger.info(s)
#
#     def save_network(self, epoch, iter_step):
#         gen_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
#         opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
#         network = self.netG
#         if isinstance(self.netG, nn.DataParallel):
#             network = network.module
#         state_dict = network.state_dict()
#         for key, param in state_dict.items():
#             state_dict[key] = param.cpu()
#         torch.save(state_dict, gen_path)
#         opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
#         opt_state['optimizer'] = self.optG.state_dict()
#         torch.save(opt_state, opt_path)
#         logger.info('Saved model in [{:s}] ...'.format(gen_path))
#
#     def load_network(self):
#         load_path = self.opt['path']['resume_state']
#         if load_path is not None:
#             logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
#             gen_path = '{}_gen.pth'.format(load_path)
#             opt_path = '{}_opt.pth'.format(load_path)
#             network = self.netG
#             if isinstance(self.netG, nn.DataParallel):
#                 network = network.module
#             network.load_state_dict(torch.load(gen_path, weights_only=True), strict=False)
#             if self.opt['phase'] == 'train' and os.path.exists(opt_path):
#                 logger.info('Found optimizer state, loading...')
#                 opt = torch.load(opt_path)
#                 self.optG.load_state_dict(opt['optimizer'])
#                 self.begin_step = opt['iter']
#                 self.begin_epoch = opt['epoch']
#             else:
#                 logger.warning('No optimizer state found at {}. Starting fine-tuning from scratch.'.format(opt_path))
#                 self.begin_step = 0
#                 self.begin_epoch = 0
