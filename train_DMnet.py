"""
作者： hsjxg
日期： 2025/7/31 16:58
"""
import torch
import dataset_ddpm
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import time
import signal  # 新增：用于捕获中断信号
import sys  # 新增：用于安全退出

# 新增：全局变量保存 diffusion 对象以便在信号处理函数中使用
global_diffusion = None
global_current_epoch = 0
global_current_step = 0

# 新增：信号处理函数，用于捕获 Ctrl+C 中断并保存模型
def signal_handler(sig, frame):
    global global_diffusion, global_current_epoch, global_current_step
    logger = logging.getLogger('base')
    if global_diffusion is not None:
        logger.info(f"Caught interrupt signal (Ctrl+C), saving model at epoch {global_current_epoch}, step {global_current_step}")
        global_diffusion.save_network(global_current_epoch, global_current_step)
        logger.info("Model and training state saved successfully.")
    else:
        logger.info("No model to save, exiting.")
    sys.exit(0)

# 新增：注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import random, numpy as np, torch

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    # # 测试代码 1：打印配置信息以确认解析正确
    # logger.info(f"Parsed config: n_iter={opt['train']['n_iter']}, "
    #             f"batch_size={opt['datasets']['train']['batch_size']}, "
    #             f"print_freq={opt['train']['print_freq']}, "
    #             f"val_freq={opt['train']['val_freq']}")

    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            # logger.info(f"Creating train dataset with dataroot: {dataset_opt['dataroot']}")
            train_set = dataset_ddpm.create_dataset(dataset_opt, phase)
            train_loader = dataset_ddpm.create_dataloader(train_set, dataset_opt, phase)
            # # 测试代码 2：检查训练数据加载
            # logger.info(f"Train dataset size: {len(train_set)}")
            # for i, train_data in enumerate(train_loader):
            #     logger.info(f"Train Batch {i}: keys={train_data.keys()}, "
            #                f"shapes={[train_data[k].shape for k in train_data.keys()]}")
            #     break  # 只检查第一批数据
        elif phase == 'val':
            # logger.info(f"Creating val dataset with dataroot: {dataset_opt['dataroot']}")
            val_set = dataset_ddpm.create_dataset(dataset_opt, phase)
            val_loader = dataset_ddpm.create_dataloader(val_set, dataset_opt, phase)
            # logger.info(f"Total samples in val_loader.dataset: {len(val_loader.dataset)}")
            # # 测试代码 3：检查验证数据加载
            # logger.info(f"Val dataset size: {len(val_set)}")
            # for i, val_data in enumerate(val_loader):
            #     logger.info(f"Val Batch {i}: keys={val_data.keys()}, "
            #                f"shapes={[val_data[k].shape for k in val_data.keys()]}")
            #     break  # 只检查第一批数据

    # model
    diffusion = Model.create_model(opt)

    # logger.info(f"NetG class: {diffusion.netG.__class__.__name__}")
    # logger.info(f"NetG attributes: {dir(diffusion.netG)}")
    # for name, module in diffusion.netG.named_modules():
    #     if 'encoder_water' in name:
    #         logger.info(f"Found encoder_water in: {name}")
    # for name, param in diffusion.netG.named_parameters():
    #     logger.info(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    # 冻结 UNet 的 encoder_water 参数
    # for param in diffusion.netG.denoise_fn.encoder_water.parameters():
    #     param.requires_grad = False
    # logger.info("UNet encoder_water parameters frozen for fine-tuning.")

    # 新增：将 diffusion 对象赋值给全局变量
    global_diffusion = diffusion
    logger.info(f"Loaded pretrained G from: {opt['path'].get('resume_state')}")
    logger.info(f"begin_epoch={diffusion.begin_epoch}, begin_step={diffusion.begin_step}")

    # # 测试代码 4：检查模型和 GPU 可用性
    # logger.info(f"Model created: {diffusion.__class__.__name__}")
    # logger.info(f"CUDA available: {torch.cuda.is_available()}, "
    #             f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # # 测试代码 5：检查 GPU 内存使用情况
    # if torch.cuda.is_available():
    #     logger.info(f"GPU Memory before training: "
    #                 f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB / "
    #                 f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            # 新增：更新全局 epoch 变量
            global_current_epoch = current_epoch
            # logger.info(f"Starting epoch {current_epoch}")
            for i, train_data in enumerate(train_loader):
                # start_time = time.time()  # 测试代码 6：记录每步时间
                current_step += 1
                # 新增：更新全局 step 变量
                global_current_step = current_step
                if current_step > n_iter:
                    break

                # # 逐步解冻 encoder_water 参数
                # if current_step == 50000:
                #     for param in diffusion.model.encoder_water.parameters():
                #         param.requires_grad = True
                #     logger.info("UNet encoder_water parameters unfrozen at step 50000.")

                # # 测试代码 7：检查数据输入
                # diffusion.feed_data(train_data)
                # logger.info(f"Step {current_step}: Data fed, shapes={[train_data[k].shape for k in train_data.keys()]}")
                diffusion.feed_data(train_data)

                # # 测试代码 8：检查优化步骤
                # diffusion.optimize_parameters()
                # logger.info(f"Step {current_step}: Optimization done, time={time.time() - start_time:.2f}s")
                diffusion.optimize_parameters()

                # # 测试代码 9：检查 GPU 内存使用
                # if torch.cuda.is_available():
                #     logger.info(f"Step {current_step}: GPU Memory: "
                #                 f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB / "
                #                 f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_uiqm = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        Enhanced_img = Metrics.tensor2img(visuals['Enhanced'])  # uint8
                        Ref_img = Metrics.tensor2img(visuals['Ref'])  # uint8
                        Raw_img = Metrics.tensor2img(visuals['Raw'])  # uint8
                        # logger.info(f"[VAL] Total validation samples: {len(val_loader.dataset)}")



                        Metrics.save_img(
                            Ref_img, '{}/{}_{}_Ref.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            Enhanced_img, '{}/{}_{}_Enhanced.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            Raw_img, '{}/{}_{}_Raw.png'.format(result_path, current_step, idx))


                        avg_psnr += Metrics.calculate_psnr(Enhanced_img, Ref_img)
                        avg_ssim += Metrics.calculate_ssim(Enhanced_img, Ref_img)
                        # avg_uiqm += Metrics.uiqm(visuals['Enhanced'])

                    avg_psnr /= idx
                    avg_ssim /= idx
                    # avg_uiqm /= idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info(
                        '# Validation # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                    logger_val = logging.getLogger('val')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()

            Ref_img = Metrics.tensor2img(visuals['Ref'])  # uint8
            Raw_img = Metrics.tensor2img(visuals['Raw'])  # uint8
            Enhanced_img = Metrics.tensor2img(visuals['Enhanced'])

            #
            # Enhanced_img_mode = 'grid'
            # if Enhanced_img_mode == 'single':
            #     Enhanced_img = visuals['Enhanced']  # uint8
            #     sample_num = Enhanced_img.shape[0]
            #     for iter in range(0, sample_num):
            #         Metrics.save_img(
            #             Metrics.tensor2img(Enhanced_img[iter]), '{}/{}_{}_Enhanced_{}.png'.format(result_path, current_step, idx, iter))
            # else:
            #     Enhanced_img = Metrics.tensor2img(visuals['Enhanced'])  # uint8
            #     Metrics.save_img(
            #         Enhanced_img, '{}/{}_{}_generation_process.png'.format(result_path, current_step, idx))
            #     Metrics.save_img(
            #         Metrics.tensor2img(visuals['Enhanced'][-1]), '{}/{}_{}_Enhanced.png'.format(result_path, current_step, idx))

            Metrics.save_img(Ref_img, f'{result_path}/{current_step}_{idx}_Ref.png')
            Metrics.save_img(Raw_img, f'{result_path}/{current_step}_{idx}_Raw.png')
            Metrics.save_img(Enhanced_img, f'{result_path}/{current_step}_{idx}_Enhanced.png')



            Metrics.save_img(
                Ref_img, '{}/{}_{}_Ref.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Raw_img, '{}/{}_{}_Raw.png'.format(result_path, current_step, idx))


            eval_psnr = Metrics.calculate_psnr(Enhanced_img, Ref_img)
            eval_ssim = Metrics.calculate_ssim(Enhanced_img, Ref_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))