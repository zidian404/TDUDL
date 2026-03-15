import os
import json
from glob import glob
from typing import Dict
import utils.utils_option as option


def generate_global_id_map(opt_path: str) -> tuple[Dict[str, int], int]:
    """生成所有split的全局id映射，按 train->valid->test 固定顺序"""
    # 读取opt
    opt = option.parse(opt_path, is_train=True)
    
    print("🔍 扫描所有数据集路径...")
    all_paths = []
    
    # 🔥 固定顺序：train -> valid -> test，确保 id 分配稳定
    split_order = ['train', 'valid', 'test']
    
    split_counts = {}
    for split_name in split_order:
        if split_name in opt:
            dataroot = opt[split_name]['dataroot_H']
            paths = sorted(glob(os.path.join(dataroot, '*')))
            count = len(paths)
            print(f"📁 {split_name}: {count} 张图，路径: {dataroot}")
            all_paths.extend(paths)
            split_counts[split_name] = count
        else:
            print(f"⚠️  {split_name} split 未在配置中")
    
    # 🔥 按 split 顺序拼接，不打乱排序
    global_paths = []
    for split_name in split_order:
        if split_name in opt:
            dataroot = opt[split_name]['dataroot_H']
            paths = sorted(glob(os.path.join(dataroot, '*')))
            global_paths.extend(paths)
    
    # 去重，但保持相对顺序
    seen = set()
    unique_paths = []
    for path in global_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    path2id = {path: idx for idx, path in enumerate(unique_paths)}
    total_n_samples = len(unique_paths)
    
    # 保存到json
    id_map = {
        'total_n_samples': total_n_samples,
        'path2id': path2id,
        'global_paths_preview': unique_paths[:5] + ['...'],  # 前5个用于验证
        'split_counts': split_counts
    }
    
    os.makedirs('utils', exist_ok=True)
    with open('utils/id_map.json', 'w') as f:
        json.dump(id_map, f, indent=2)
    
    print(f"✅ 生成全局id映射: {total_n_samples} 张图")
    print(f"📄 保存至: utils/id_map.json")
    print(f"📊 split_counts: {split_counts}")
    return path2id, total_n_samples


def load_global_id_map() -> tuple[Dict[str, int], int]:
    """加载预生成的id映射"""
    try:
        with open('utils/id_map.json', 'r') as f:
            id_map = json.load(f)
        path2id = id_map['path2id']
        n_samples = id_map['total_n_samples']
        print(f"✅ 加载全局id映射: {n_samples} 张图")
        return path2id, n_samples
    except FileNotFoundError:
        raise RuntimeError("❌ utils/id_map.json 不存在，请先运行 generate_global_id_map('./options/train_options.json')")
