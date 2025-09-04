import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms

from model.UnderwaterRestorationNetV3 import PGLANPlus
from datasets_PGLAN.UnderwaterDataset import UnderwaterDataset
from losses.losses import CombinedLoss
from utils.metrics import compute_metrics


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# 训练
def train_one_epoch(model, dataloader, optimizer, device, criterion, writer, epoch):
    model.train()
    total_loss = 0
    total_loss_details = {}

    # 使用tqdm来显示一个epoch的进度条，但不再在循环内部打印
    for i, (input_img, gt_img) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        input_img = input_img.to(device)
        gt_img = gt_img.to(device)

        optimizer.zero_grad()
        # 假设model的输出是J_hat, T, A
        # 如果模型输出不是这三个，请根据你的模型进行调整
        J_hat, T, A = model(input_img)

        loss, loss_details = criterion(input_img, J_hat, T, A, gt_img)
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 累加总损失
        total_loss += loss.item()

        # 累加每个子损失项
        for key, value in loss_details.items():
            if key not in total_loss_details:
                total_loss_details[key] = 0
            total_loss_details[key] += value

    # 计算每个epoch的平均总损失和平均子损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_details = {k: v / len(dataloader) for k, v in total_loss_details.items()}

    # 在TensorBoard中记录每个epoch的平均损失
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    for key, value in avg_loss_details.items():
        writer.add_scalar(f"DetailedLoss/{key}_epoch", value, epoch)

    # 返回平均损失，用于后续的保存和检查点
    return avg_loss, avg_loss_details


def validate(model, dataloader, device, writer, epoch):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for input_img, gt_img in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            input_img = input_img.to(device)
            gt_img = gt_img.to(device)

            J_hat, _, _ = model(input_img)
            metrics = compute_metrics(J_hat, gt_img)
            all_metrics.append(metrics)

    avg_metrics = {
        k: sum([m[k] for m in all_metrics]) / len(all_metrics)
        for k in all_metrics[0].keys()
    }

    for k, v in avg_metrics.items():
        writer.add_scalar(f"Metrics/{k}", v, epoch)

    return avg_metrics


def main():
    config = load_config("config/PGLAN.json")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config["save_path"], exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(config["save_path"], "runs"))

    transform = transforms.Compose([
        transforms.Resize(tuple(config["resize"])),
        transforms.ToTensor(),
    ])

    train_dataset = UnderwaterDataset(config["train_input"], config["train_gt"], transform)
    val_dataset = UnderwaterDataset(config["val_input"], config["val_gt"], transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ✅ 使用终极模型
    model = PGLANPlus(base_channels=config["base_channels"]).to(device)

    # ✅ 优化的学习率和优化器
    initial_lr = config["learning_rate"]
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    print(f"🔧 Using learning rate: {initial_lr}")

    # ✅ 基于性能的学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10,
        verbose=True, min_lr=1e-7
    )

    # ✅ 简单高效的损失函数
    criterion = CombinedLoss(device=device)

    start_epoch = 1
    best_psnr = 0

    # 检查点加载
    if config.get("resume", False) and os.path.exists(config["resume_path"]):
        print(f"Resuming from checkpoint: {config['resume_path']}")
        checkpoint = torch.load(config["resume_path"])

        try:
            model.load_state_dict(checkpoint["model"], strict=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Checkpoint loading issue: {e}")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint["model"].items()
                               if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded {len(pretrained_dict)} compatible parameters")

        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", 0)

        # 重置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        print(f"🔧 Reset learning rate to: {initial_lr}")
    else:
        print("Training from scratch.")

    # 训练历史记录
    psnr_history = []

    for epoch in range(start_epoch, config["epochs"] + 1):
        print(f"\n{'=' * 60}")
        print(f"🚀 Epoch {epoch}/{config['epochs']}")
        print(f"{'=' * 60}")

        # 训练
        train_loss, train_loss_details = train_one_epoch(
            model, train_loader, optimizer, device, criterion, writer, epoch
        )

        # 验证
        val_metrics = validate(model, val_loader, device, writer, epoch)

        # 学习率调度（基于PSNR）
        scheduler.step(val_metrics['PSNR'])

        # 记录历史
        psnr_history.append(val_metrics['PSNR'])

        # 显示结果
        print(f"\n📈 Training Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"\n📊 Detailed Train Losses:")
        for key, value in train_loss_details.items():
            print(f"   {key}: {value:.4f}")

        print(f"\n📊 Validation Metrics:")
        print(f"   PSNR: {val_metrics['PSNR']:.6f}")
        print(f"   SSIM: {val_metrics['SSIM']:.6f}")

        # 显示改进趋势
        if len(psnr_history) >= 5:
            recent_trend = psnr_history[-1] - psnr_history[-5]
            trend_symbol = "↗️" if recent_trend > 0 else "↘️" if recent_trend < 0 else "➡️"
            print(f"   Recent 5-epoch trend: {recent_trend:+.4f} {trend_symbol}")

        print(f"\n⚙️ Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.8f}")

        # 保存最佳模型
        if val_metrics['PSNR'] > best_psnr:
            best_psnr = val_metrics['PSNR']
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr
            }, os.path.join(config["save_path"], "best_model.pth"))
            print(f"🏆 New best PSNR: {best_psnr:.6f}, model saved!")

        # 保存定期检查点
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr
        }, os.path.join(config["save_path"], "latest.pth"))

        if epoch % config.get("save_interval", 10) == 0:
            torch.save(model.state_dict(), os.path.join(config["save_path"], f"model_epoch_{epoch}.pth"))
            print(f"💾 Model saved at epoch {epoch}")

        print(f"{'=' * 60}")

    writer.close()
    print(f"\n🎉 Training completed! Best PSNR: {best_psnr:.6f}")


if __name__ == '__main__':
    main()
