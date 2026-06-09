import torch.utils.data as data
import os
import torch
from typing import List
import numpy as np
from utils import utils_image as util
import random
from copy import deepcopy
from glob import glob

class dataset_admm_denose(data.Dataset):
    def __init__(self, opt, task, image_path=None, label_idx=None):
        self.opt = opt
        self.task = task
        self.n_channels = opt['n_channels'] # 此时为 1 (纯灰度)
        
        if task == 'train':
            # 💥 分类模式：不仅获取图片路径，同时动态分析文件夹名作为分类标签
            # 期望路径：dataroot_H/类别子文件夹/*.jpg
            root_dir = self.opt['dataroot_H']
            
            # 1. 自动扫描子目录获取所有类别名称并排序，确保标签固定映射 (例如: {'agi_pelikano': 0, 'category2': 1})
            self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
            
            self.img_paths = []
            self.labels = []
            
            # 2. 收集所有图片的路径以及它对应的分类数字标签
            for cls_name in self.class_names:
                cls_dir = os.path.join(root_dir, cls_name)
                # 适配各种常见图片格式
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
                    found_files = glob(os.path.join(cls_dir, ext))
                    for f_path in found_files:
                        self.img_paths.append(f_path)
                        self.labels.append(self.class_to_idx[cls_name])
            
            self.sigma = opt.get('sigma', [0])
            if 'H_size' in opt:
                self.patch_size = opt['H_size'] # 128
        else:
            # 💥 验证/测试模式：单张图片加载，直接通过外部传入当前图片的分类标签
            self.img_paths = [image_path] if image_path else []
            self.labels = [label_idx] if label_idx is not None else [0]
            self.sigma = opt.get('sigma', [0])

    def __getitem__(self, index):
        # 1. 安全读取纯灰度织物图像
        img_path = self.img_paths[index]
        class_id = self.labels[index]
        
        # 针对 Windows 环境下的中文路径采用安全读取（防止因特殊字符报错）
        try:
            import cv2
            img_H = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img_H is not None:
                # 补回通道轴，确保维度为 (H, W, 1) 适配后面的工具函数
                img_H = np.expand_dims(img_H, axis=2)
            else:
                img_H = util.imread_uint(img_path, self.n_channels)
        except:
            img_H = util.imread_uint(img_path, self.n_channels)

        H, W = img_H.shape[:2]
        
        if self.task == 'train':
            # 2. 训练阶段：随机裁剪 (128x128)
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # 3. 数据增强（随机旋转、镜像，提升泛化能力）
            patch_H = util.augment_img(patch_H, mode=np.random.randint(0, 8))

            # HWC to CHW, tensor 转换
            img_L = util.uint2tensor3(patch_H)
            
            # 4. 噪声电平控制：既然变为纯分类模型且输入噪声为0，则固定将 sigma 设为 0
            noise_value = 0.0
            noise_level = torch.FloatTensor([noise_value]) / 255.0

        else:
            # 5. 验证阶段：处理全量大图
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # 保持纯净的无噪声分类状态
            sigma_value = 0.0
            noise_level = torch.FloatTensor([sigma_value / 255.0])

            img_L = util.single2tensor3(img_L)
            
            # 缩减为 8 的倍数尺寸，防止 Restormer 在下采样时出现尺寸对齐异常
            h, w = img_L.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, (w // 8 * 8))
            img_L = img_L[..., top, left]

        # 💥 最终返回：[输入灰度图张量, 对应分类标签数字, 噪声等级张量]
        # 这与你更新后的 train原WS.py 中的：img_L, labels, noise_level = batch_data 完美对齐
        return img_L, class_id, noise_level

    def __len__(self):
        return len(self.img_paths)


def get_data(opt, task):
    if task == 'train':
        opt_ = opt[task]
        dataset = dataset_admm_denose(opt_, task)
        return dataset
    else:
        # 💥 验证/测试模式：为每一个类别的每张图片分配对应的真实标签
        datasets: List[dataset_admm_denose] = []
        opt_ = opt[task]
        root_dir = opt_['dataroot_H']
        
        # 提取验证集包含的类别，保证与训练集映射字典绝对同步一致
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        
        # 遍历所有类别的文件夹，生成独立的子测试加载器
        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            paths = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
                paths.extend(glob(os.path.join(cls_dir, ext)))
                
            paths = sorted(paths)
            label_idx = class_to_idx[cls_name]
            
            for path in paths:
                opt_subset = deepcopy(opt_) 
                datasets.append(dataset_admm_denose(
                    opt_subset, 
                    task, 
                    image_path=path,        # 传递当前图片路径
                    label_idx=label_idx     # 传递当前图片对应的真实类别数字标签
                ))

        return datasets