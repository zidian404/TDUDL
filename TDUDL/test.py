from typing import Dict, List
import torch.utils.data as data
import torch
import time
import os
import logging
from glob import glob
from prettytable import PrettyTable
import numpy as np
from collections import OrderedDict
import Net.denoise_net as net  
from utils.dataset_admm import get_data  
import utils.utils_option as option
import utils.utils_image as image
from utils import utils_logger
from utils.global_id_map import load_global_id_map, generate_global_id_map

# 构建全局id映射
def build_global_id_mapping_test(opt: Dict) -> tuple[Dict[str, int], int]:
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
    return path2id, total_n_samples

def safe_forward(model, img_L, noise_level, ids):
    with torch.no_grad():
        if torch.isnan(img_L).any() or torch.isinf(img_L).any():
            img_L = torch.nan_to_num(img_L, nan=0.0, posinf=1.0, neginf=0.0)
        
        test_out, _ = model(img_L, noise_level, ids)
        
        if torch.isnan(test_out).any() or torch.isinf(test_out).any():
            test_out = torch.nan_to_num(test_out, nan=0.0, posinf=1.0, neginf=0.0)
        return test_out

if __name__ == '__main__':
    gpus = '0'  
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)
    
    logger_name = 'test_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    try:
        path2id, total_n_samples = load_global_id_map()
        print(f"📈 从id_map.json加载映射: 总样本数={total_n_samples}")
    except:
        path2id, total_n_samples = build_global_id_mapping_test(opt)

    test_data_path = opt['test']['dataroot_H']
    img_names = sorted([os.path.basename(name) for name in glob(os.path.join(test_data_path, '*'))])
    test_set_list = get_data(opt, 'test', path2id=path2id) 
    
    sigma_list = opt['test']['sigma']
    sigma_size = len(sigma_list)

    model = net.denoise_Net_admm_restormer(opt, n_samples=total_n_samples)
    model.to(device)
    
    pretrained_path = opt['pretained_path']
    print(f"Loading model from {pretrained_path}...")
    state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model_dict = model.state_dict()
    for k, v in new_state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k].copy_(v)
    model.load_state_dict(model_dict, strict=True)
    model.eval()

    detailed_psnrs = OrderedDict()
    detailed_ssims = OrderedDict()

    print("======================= 测试开始 =======================")
    
    for s_idx, sigma_level in enumerate(sigma_list):
        current_sigma_dataset = test_set_list[s_idx]
        loader = data.DataLoader(current_sigma_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"\n🚀 正在测试 Sigma = {sigma_level} ...")
        
        for i, batch_data in enumerate(loader):
            img_H, img_L, noise_level, ids = batch_data
            img_name = img_names[i]
            
            img_L, noise_level, ids = img_L.to(device), noise_level.to(device), ids.to(device)

            with torch.no_grad():
                output = safe_forward(model, img_L, noise_level, ids)
            
            out_np = image.tensor2uint(output)
            gt_np = image.tensor2uint(img_H)
            psnr = image.calculate_psnr(out_np, gt_np)
            ssim = image.calculate_ssim(out_np, gt_np)

            if img_name not in detailed_psnrs:
                detailed_psnrs[img_name] = [0.0] * sigma_size
                detailed_ssims[img_name] = [0.0] * sigma_size
            
            detailed_psnrs[img_name][s_idx] = psnr
            detailed_ssims[img_name][s_idx] = ssim
            
            print(f"  [{i+1}/{len(img_names)}] Image: {img_name} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

    # --- 修改部分：结果展示与多指标表格生成 ---
    print("\n" + "="*30 + " 统计报表 " + "="*30)
    
    # 构建表头：Image Name | σ=15 (P/S) | σ=25 (P/S) ...
    header = ['Image Name'] + [f"σ={s} (PSNR/SSIM)" for s in sigma_list]
    
    # 1. 生成单图明细表 (PSNR / SSIM 同列显示)
    t_detail = PrettyTable(header)
    for name in img_names:
        row = [name]
        for s_idx in range(sigma_size):
            p = detailed_psnrs[name][s_idx]
            s = detailed_ssims[name][s_idx]
            row.append(f"{p:.2f} / {s:.4f}")
        t_detail.add_row(row)
    
    # 2. 生成汇总平均表 (PSNR 和 SSIM 分行显示，看得更清楚)
    # 汇总表头：Metric | σ=15 | σ=25 | σ=50
    summary_header = ['Metric'] + [f"σ={s}" for s in sigma_list]
    t_summary = PrettyTable(summary_header)
    
    avg_psnr_row = ["Average PSNR"]
    avg_ssim_row = ["Average SSIM"]
    
    for s_idx in range(sigma_size):
        all_p = [detailed_psnrs[name][s_idx] for name in img_names]
        all_s = [detailed_ssims[name][s_idx] for name in img_names]
        avg_psnr_row.append(f"{np.mean(all_p):.2f}")
        avg_ssim_row.append(f"{np.mean(all_s):.4f}")
    
    t_summary.add_row(avg_psnr_row)
    t_summary.add_row(avg_ssim_row)

    # 打印到控制台
    print("\n[明细表] 每张图片在不同噪声水平下的指标 (PSNR / SSIM):")
    print(t_detail)
    print("\n[汇总表] 数据集整体平均指标:")
    print(t_summary)

    # 写入日志 (确保 SSIM 被记录)
    logger.info(f"\nDetailed Test Results (PSNR / SSIM):\n{t_detail}")
    logger.info(f"\nFinal Average Summary:\n{t_summary}")
    
    print(f"\n✅ 测试完成！详细指标已存入日志: {opt['log_path']}")