import os
import random
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.utils.data as data

from utils import utils_image as util  # 你工程里本来就是这个名字

class dataset_admm_denose(data.Dataset):
    def __init__(self, opt_split: Dict, split: str, path2id: Dict[str, int] = None):
        """
        新增参数：path2id - 全局路径到唯一id的映射
        """
        super().__init__()
        self.split = split
        self.opt_split = opt_split
        self.n_channels = opt_split['n_channels']
        self.path2id = path2id  # 🔥 新增：全局id映射

        # 统一：按文件名排序
        dataroot = opt_split['dataroot_H']
        self.img_paths: List[str] = sorted(glob(os.path.join(dataroot, '*')))
        if len(self.img_paths) == 0:
            raise RuntimeError(f'No images found in {dataroot}')

        # 🔥 修改：使用全局唯一id
        if self.path2id is not None:
            self.img_ids: List[int] = []
            for img_path in self.img_paths:
                if img_path not in self.path2id:
                    raise ValueError(f"路径 {img_path} 不在全局映射中")
                self.img_ids.append(self.path2id[img_path])
            print(f"📈 {split} 数据集使用全局id: {min(self.img_ids)} ~ {max(self.img_ids)}")
        else:
            # 兼容旧版本
            self.img_ids: List[int] = list(range(len(self.img_paths)))

        # 以下不变...
        if split == 'train':
            self.sigma_range = opt_split['sigma']
            assert len(self.sigma_range) == 2, \
                f"train.sigma 应该是 [min,max] 两个数, 当前: {self.sigma_range}"
            self.patch_size = opt_split.get('H_size', 128)
        elif split in ['valid', 'test']:
            self.sigma_list: List[float] = [float(s) for s in opt_split['sigma']]
            self.pairs: List[Tuple[int, float]] = []
            for img_idx, _ in enumerate(self.img_paths):
                for s in self.sigma_list:
                    self.pairs.append((img_idx, float(s)))
        else:
            raise ValueError(f'Unknown split: {split}')

    def __len__(self) -> int:
        if self.split == 'train':
            # 一个 epoch 内每张图取一次随机 patch
            return len(self.img_paths)
        else:
            # valid/test: 所有 (img, sigma) 组合
            return len(self.pairs)

    def _read_img_as_tensor(self, img_path: str) -> torch.Tensor:
        """
        读图并转为 [C,H,W]、[0,1] float tensor，与原工程保持一致
        """
        img_uint = util.imread_uint(img_path, self.n_channels)  # HWC, uint8
        img_tensor = util.uint2tensor3(img_uint)  # [C,H,W], float, [0,1]
        return img_tensor

    def __getitem__(self, index: int):
        if self.split == 'train':
            # ===================== 训练 =====================
            # 1. 选一张原图
            img_path = self.img_paths[index]
            img_id = self.img_ids[index]

            # 2. 读原图
            img_H_uint = util.imread_uint(img_path, self.n_channels)  # HWC, uint8
            H, W = img_H_uint.shape[:2]
            ps = self.patch_size

            # 3. 随机裁剪 patch
            if H < ps or W < ps:
                # 如果原图比 patch 小，先简单 pad 一下
                pad_h = max(0, ps - H)
                pad_w = max(0, ps - W)
                img_H_uint = np.pad(img_H_uint,
                                  ((0, pad_h), (0, pad_w), (0, 0)),
                                  mode='reflect')
                H, W = img_H_uint.shape[:2]

            rnd_h = random.randint(0, H - ps)
            rnd_w = random.randint(0, W - ps)
            patch_H = img_H_uint[rnd_h:rnd_h + ps, rnd_w:rnd_w + ps, :]

            # 4. 随机增强
            patch_H = util.augment_img(patch_H, mode=np.random.randint(0, 8))

            # 5. 转 tensor
            img_H = util.uint2tensor3(patch_H)  # [C,H,W], [0,1]
            img_L = img_H.clone()

            # 6. 随机采样噪声 sigma (在区间内均匀)
            sigma = float(np.random.uniform(self.sigma_range[0],
                                          self.sigma_range[1]))
            noise_level = torch.FloatTensor([sigma / 255.0])  # [1]

            # 7. 在 tensor 上加高斯噪声（与 valid/test 保持一致）
            noise = torch.randn_like(img_L).mul_(noise_level.view(-1, 1, 1))
            img_L = img_L + noise

            # 🔥 修复：返回 scalar long，而不是 [1]
            img_id = torch.tensor(img_id, dtype=torch.long)  # scalar long

            return img_H, img_L, noise_level, img_id

        else:
            # ===================== 验证 / 测试 =====================
            # index 对应 (img_idx, sigma)
            img_idx, sigma = self.pairs[index]
            img_path = self.img_paths[img_idx]
            img_id = self.img_ids[img_idx]

            # 读原图 -> tensor（**直接整图，原尺寸！**）
            img_H = self._read_img_as_tensor(img_path)  # [C,H_orig,W_orig]
            img_L = img_H.clone()

            # 固定噪声 sigma
            noise_level = torch.FloatTensor([sigma / 255.0])

            # 在 tensor 上加噪（和训练完全一致）
            noise = torch.randn_like(img_L).mul_(noise_level.view(-1, 1, 1))
            img_L = img_L + noise

            # 🔥 修复：返回 scalar long，而不是 [1]
            img_id = torch.tensor(img_id, dtype=torch.long)  # scalar long

            return img_H, img_L, noise_level, img_id

def get_data(opt_all: dict, task: str, path2id: Dict[str, int] = None):
    """
    新增参数：path2id - 全局映射
    """
    if task == 'train':
        opt_split = opt_all['train']
        dataset = dataset_admm_denose(opt_split, split='train', path2id=path2id)
        return dataset

    elif task in ['valid', 'test']:
        opt_split = opt_all[task]
        sigma_list = [float(s) for s in opt_split['sigma']]

        datasets: List[dataset_admm_denose] = []
        for sigma in sigma_list:
            opt_one = dict(opt_split)  # 浅拷贝
            opt_one['sigma'] = [sigma]  # 单元素列表
            datasets.append(dataset_admm_denose(opt_one, split=task, path2id=path2id))
        return datasets

    else:
        raise ValueError(f'Unknown task for get_data: {task}')