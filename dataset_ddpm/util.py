import os
import torch
import torchvision
import random
import numpy as np
from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    # assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [totensor(img) for img in img_list]
#     if split == 'train':
#         imgs = torch.stack(imgs, 0)
#         imgs = hflip(imgs)
#         imgs = torch.unbind(imgs, dim=0)
#     ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
#     return ret_img
def transform_augment(img_list, split='val', min_max=(0, 1), size=256):
    # 统一 resize 所有图像
    img_list = [img.resize((size, size), Image.BICUBIC) for img in img_list]

    # 转 tensor
    imgs = [totensor(img) for img in img_list]

    # 数据增强
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)

    # 归一化
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


# from torchvision import transforms
# def transform_augment(img_list, split='val', min_max=(-1, 1), image_size=256):
#     # 定义变换
#     transform_list = [
#         transforms.Resize((image_size, image_size)),  # 统一调整为 image_size x image_size
#         transforms.ToTensor(),  # 转换为张量
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
#     ]
#     if split == 'train':
#         transform_list.insert(0, transforms.RandomCrop(image_size))  # 随机裁剪
#         transform_list.insert(0, transforms.RandomHorizontalFlip())  # 随机水平翻转
#
#     transform = transforms.Compose(transform_list)
#     imgs = [transform(img) for img in img_list]
#
#     # 调整到指定 min_max 范围（与 LRHRDataset/LRHRDataset2 的 (-1, 1) 一致）
#     ret_img = [img * (min_max[1] - min_max[0]) / 2 + (min_max[1] + min_max[0]) / 2 for img in imgs]
#     return ret_img
