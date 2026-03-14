import os
import json
from glob import glob
from typing import Dict

def generate_global_id_map(opt_path: str) -> tuple[Dict[str, int], int]:
    """一次性生成所有split的全局id映射，保存为json"""
    # 读取opt（假设有train_options.json）
    import utils.utils_option as option
    opt = option.parse(opt_path, is_train=True)
    
    all_paths = []
    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        if split_name in opt:
            dataroot = opt[split_name]['dataroot_H']
            paths = sorted(glob(os.path.join(dataroot, '*')))
            all_paths.extend(paths)
    
    global_paths = sorted(list(set(all_paths)))  # 去重+排序
    path2id = {path: idx for idx, path in enumerate(global_paths)}
    
    # 保存到json
    id_map = {
        'total_n_samples': len(global_paths),
        'path2id': path2id,
        'global_paths': global_paths[:10] + ['...'],  # 前10个用于验证
        'split_counts': {split_name: len(opt.get(split_name, {}).get('dataroot_H', [])) for split_name in split_names}
    }
    
    with open('utils/id_map.json', 'w') as f:
        json.dump(id_map, f, indent=2)
    
    print(f"✅ 生成全局id映射: {len(global_paths)} 张图")
    print(f"📄 保存至: utils/id_map.json")
    return path2id, len(global_paths)

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
        raise RuntimeError("请先运行 generate_global_id_map 生成 id_map.json")
