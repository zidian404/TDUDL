import os
from glob import glob
from typing import List

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from utils import utils_image as util


IMG_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')


class dataset_admm_denose(data.Dataset):
    def __init__(self, opt, task):
        """
        统一的分类数据集：
        - train:  RandomResizedCrop(128) + RandomHorizontalFlip
        - valid:  Resize(resize_size) + CenterCrop(128)
        返回: img_L (1x128x128), class_id (int), noise_level (1,)
        """
        self.opt = opt
        self.task = task
        self.n_channels = opt.get('n_channels', 1)
        self.img_size = opt.get('H_size', 128)
        self.resize_size = opt.get('resize_size', 146)

        root_dir = self.opt['dataroot_H']

        # 类别名 = 子文件夹名
        self.class_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.img_paths = []
        self.labels = []

        for cls_name in self.class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for ext in IMG_EXTENSIONS:
                found_files = sorted(glob(os.path.join(cls_dir, ext)))
                for f_path in found_files:
                    self.img_paths.append(f_path)
                    self.labels.append(self.class_to_idx[cls_name])

        # 训练增强：YOLO 风格随机裁剪 + 随机水平翻转
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),  # (H, W) -> PIL Image (L)
            transforms.RandomResizedCrop(
                size=self.img_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()      # -> [1, 128, 128], 0~1
        ])

        # 验证增强：Resize + CenterCrop（固定视野）
        self.valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        class_id = self.labels[index]

        # 安全读取灰度图
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("cv2 read failed")
        except Exception:
            # 回退到原来的 util 读取
            img = util.imread_uint(img_path, 1)[:, :, 0]  # (H, W, 1) -> (H, W)

        if self.task == 'train':
            img_L = self.train_transform(img)
        else:
            img_L = self.valid_transform(img)

        # 噪声等级固定为 0（分类任务）
        noise_level = torch.zeros(1, dtype=torch.float32)

        return img_L, class_id, noise_level

    def __len__(self):
        return len(self.img_paths)


def get_data(opt, task):
    """
    注意：现在 train 和 valid 都返回单个 Dataset，
    不再返回 Dataset 列表（test_loaders 那套田字格逻辑不再使用）。
    """
    opt_ = opt[task]
    dataset = dataset_admm_denose(opt_, task)
    return dataset