from typing import Dict, List
import torch.utils.data as data
import torch
import time
import os
import logging
from glob import glob
from prettytable import PrettyTable
from torch import cuda
import numpy as np
from collections import OrderedDict  # ← **新增这行！**
import Net.denoise_net as net  # 与train.py保持一致
from utils.dataset_admm import get_data  # 使用新的dataset_admm-3.py
import utils.utils_option as option
import utils.utils_image as image
from utils import utils_logger
from utils.global_id_map import load_global_id_map, generate_global_id_map  # 🔥 全局id映射

# 🔥 在主函数开头添加全局id映射（和train一致）
def build_global_id_mapping_test(opt: Dict) -> tuple[Dict[str, int], int]:
    """测试时构建全局映射（包含test）"""
    all_paths = []
    for split_name in ['train', 'valid', 'test']:
        if split_name in opt:
            split_opt = opt[split_name]
            dataroot = split_opt['dataroot_H']
            paths = sorted(glob(os.path.join(dataroot, '*')))
            all_paths.extend(paths)
    
    global_paths = sorted(list(set(all_paths)))
    path2id = {path: idx for idx, path in enumerate(global_paths)}
    total_n_samples = len(global_paths)
    
    print(f"🔍 测试全局图库: {total_n_samples} 张图")
    return path2id, total_n_samples

try:
    path2id, total_n_samples = load_global_id_map()
except FileNotFoundError:
    print("⚠️  id_map.json 不存在，请先运行 train.py 生成")
    path2id, total_n_samples = build_global_id_mapping_test(opt)  # fallback到你原来的函数



def safe_forward(model, img_L, noise_level, ids):
    """安全的前向传播，处理NaN/Inf"""
    with torch.no_grad():
        if torch.isnan(img_L).any() or torch.isinf(img_L).any():
            img_L = torch.nan_to_num(img_L, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(noise_level).any() or torch.isinf(noise_level).any():
            noise_level = torch.nan_to_num(noise_level, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(ids).any() or torch.isinf(ids).any():
            ids = torch.nan_to_num(ids, nan=0.0, posinf=1.0, neginf=0.0)
        
        test_out, _ = model(img_L, noise_level, ids)
        if torch.isnan(test_out).any() or torch.isinf(test_out).any():
            test_out = torch.nan_to_num(test_out, nan=0.0, posinf=1.0, neginf=0.0)
        return test_out

if __name__ == '__main__':
    # GPU设置
    gpus = '0'  # 与train.py一致
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 随机种子
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("======================= 测试开始 =======================")

    # 配置加载
    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)
    
    # 日志
    logger_name = 'test_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # 测试图像名称（用于显示）
    test_data_path = opt['test']['dataroot_H']
    names = sorted([os.path.basename(name) for name in glob(os.path.join(test_data_path, '*'))])
    print(f"Found {len(names)} test images in path {test_data_path}")

    print("Loading test datasets...")
    test_set = get_data(opt, 'test', path2id=path2id)  # 返回 List[Dataset]，每个sigma一个
    print(f"Loaded {len(test_set)} test sets (one for each sigma)")

    # 创建DataLoader
    test_loaders: List[data.DataLoader] = []
    for valid in test_set:
        loader = data.DataLoader(
            dataset=valid, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,  # 大图建议用0，避免多进程问题
            drop_last=False,
            pin_memory=True
        )
        test_loaders.append(loader)
    print(f"Total {len(test_loaders)} DataLoaders created.")

    # 模型加载
    print("Loading model...")
    model = net.denoise_Net_admm_restormer(opt, n_samples=total_n_samples)  # 与train.py一致
    model.to(device)
    
    pretrained_path = opt['pretained_path']
    if not os.path.exists(pretrained_path):
        print(f"ERROR: Model file not found: {pretrained_path}")
        exit(1)
    
    print(f"Loading pretrained model from {pretrained_path}")
    state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state
    
    # 处理DataParallel权重（现在可以用 OrderedDict 了）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Model loaded successfully!")
    model.eval()

    # 测试统计
    avg_psnrs: Dict[str, List[float]] = {}
    avg_ssims: Dict[str, List[float]] = {}
    sigma_size = len(opt['test']['sigma'])
    total_inference_time = 0
    total_batches = 0

    print(f"Full testing with {len(test_loaders)} loaders...")
    print("======================= 测试进行中 =======================")

    for loader_idx, test_loader in enumerate(test_loaders):
        avg_psnr = 0.0
        avg_ssim = 0.0
        
        # 数据集名称和sigma
        image_index = loader_idx // sigma_size
        sigma_idx = loader_idx % sigma_size
        sigma_level = opt['test']['sigma'][sigma_idx]
        dataset_name = names[image_index] if image_index < len(names) else f"UnknownDataset_{image_index}"
        
        print(f"\n- Processing {dataset_name}, sigma={sigma_level}")
        loader_inference_time = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch_count += 1
                
                # 统一解包4元组，与dataset_admm.py一致
                img_H, img_L, noise_level, ids = batch_data  # ids现在来自dataset
                
                # 移动到设备（大图优化）
                img_H = img_H.to(device, non_blocking=True)
                img_L = img_L.to(device, non_blocking=True)
                noise_level = noise_level.to(device, non_blocking=True)
                ids = ids.to(device, non_blocking=True)
                
                # 前向传播
                start_time = time.time()
                test_out = safe_forward(model, img_L, noise_level, ids)
                batch_time = time.time() - start_time
                
                total_inference_time += batch_time
                loader_inference_time += batch_time
                total_batches += 1
                
                # 处理维度（与train.py验证部分一致）
                if isinstance(test_out, (list, tuple)):
                    test_out = test_out[0]
                if test_out.dim() == 3:
                    test_out = test_out.unsqueeze(1)
                if img_H.dim() == 3:
                    img_H = img_H.unsqueeze(1)
                
                # 尺寸对齐（支持整图）
                if test_out.shape[2:] != img_H.shape[2:]:
                    test_out = torch.nn.functional.interpolate(
                        test_out, size=img_H.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # 计算指标
                test_out_np = image.tensor2uint(test_out)
                img_H_np = image.tensor2uint(img_H)
                psnr = image.calculate_psnr(test_out_np, img_H_np)
                ssim = image.calculate_ssim(test_out_np, img_H_np)
                
                avg_psnr += psnr
                avg_ssim += ssim
                
                print(f"  Batch {batch_idx+1}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
        
        if batch_count > 0:
            avg_psnr = round(avg_psnr / batch_count, 2)
            avg_ssim = round(avg_ssim * 100 / batch_count, 2)
            avg_time = loader_inference_time / batch_count
            print(f"Completed {dataset_name} (sigma={sigma_level}): PSNR={avg_psnr}, SSIM={avg_ssim}, Time={avg_time:.4f}s per batch/image")
            print("--------------------------------------------------")
        
        # 按数据集累积结果
        if dataset_name not in avg_psnrs:
            avg_psnrs[dataset_name] = [0] * sigma_size
            avg_ssims[dataset_name] = [0] * sigma_size
        avg_psnrs[dataset_name][sigma_idx] = avg_psnr
        avg_ssims[dataset_name][sigma_idx] = avg_ssim

    # 总体推理时间
    if total_batches > 0:
        avg_inference_time = total_inference_time / total_batches
        logger.info(f"Average inference time per batch/image: {avg_inference_time:.4f} s")
        print(f"Average inference time per batch/image: {avg_inference_time:.4f} s")

    # 按sigma展示结果表格
    print("\n======================= 测试结果 =======================")
    header = ['Dataset'] + [f"σ={s}" for s in opt['test']['sigma']]
    
    tpsnr = PrettyTable(header)
    for key, value in avg_psnrs.items():
        tpsnr.add_row([key] + value)
    
    tssim = PrettyTable(header)
    for key, value in avg_ssims.items():
        tssim.add_row([key] + value)
    
    logger.info(f"Test PSNR:\n{tpsnr}")
    logger.info(f"Test SSIM:\n{tssim}")
    print("Test PSNR:\n" + str(tpsnr))
    print("Test SSIM:\n" + str(tssim))

    print(f"\n测试完成！处理了 {total_batches} 个批次，{len(avg_psnrs)} 个数据集。")
