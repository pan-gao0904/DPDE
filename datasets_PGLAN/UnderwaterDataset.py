# 代码内容：
"""
作者： hsjxg
日期： 2025/8/5 18:13
"""
import os
from PIL import Image
from torch.utils.data import Dataset


class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform

        # 读取所有图像文件名（假设 input 和 gt 文件名一致）
        self.filenames = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_path = os.path.join(self.input_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        # 读取图像
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)

        return input_img, gt_img
