from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import dataset_ddpm.util as Util
import numpy as np
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO,  # 设置日志级别为 INFO
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志输出格式
                    handlers=[logging.StreamHandler()])  # 输出到控制台
logger = logging.getLogger('base')

class Conditiondataset2(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=256, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            self.style_path = Util.get_paths_from_images(
                '{}/style_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            # logger.info(f"Total dataset length: {self.dataset_len}")
            # logger.info(f"HR images: {len(self.hr_path)}")
            # logger.info(f"SR images: {len(self.sr_path)}")
            # logger.info(f"dataset_len: {self.dataset_len}, data_len: {self.data_len}")
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        # img_style = Image.open(self.style_path[index]).convert("RGB")
        img_style = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            img_LR = Image.open(self.lr_path[index]).convert("RGB")
        # if self.need_LR:
            [img_LR, img_SR, img_HR, img_style] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_style], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'Ref': img_HR, 'Raw': img_SR, 'Index': index}
        else:
            [img_SR, img_HR, img_style] = Util.transform_augment(
                [img_SR, img_HR, img_style], split=self.split, min_max=(-1, 1))
            return {'Ref': img_HR, 'Raw': img_SR,  'Index': index}
