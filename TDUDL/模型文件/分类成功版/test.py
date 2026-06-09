from typing import Dict, List
import torch.utils.data as data
import torch
import time
import os
import logging
from glob import glob
from prettytable import PrettyTable
import numpy as np
import random
import matplotlib.pyplot as plt

# 模型与工具包导入
import Net.denoise_net as net
from utils.dataset_admm import get_data
import utils.utils_option as option
from utils import utils_logger

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """ 计算 Top-1 和 Top-5 命中数量 """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res

if __name__ == '__main__':
    # 1. 硬件环境与随机种子设置
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
    
    seed_ = 1234
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    
    # 2. 解析配置文件路径
    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)
    
    # 初始化学术评测专用日志
    logger_name = 'test_cls_eval_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    print(f"Using GPU for Textile End-to-End Comprehensive Classification Evaluation!\n")

    # 3. 载入数据集 (全量大图模式)
    print("加载 Motif-Sikka-ROI 验证/测试数据集...")
    names = []
    test_data_path = opt['test']['dataroot_H']
    for name in sorted(glob(os.path.join(test_data_path, '*'))):
        names.append(os.path.basename(name))
        
    test_set = get_data(opt, 'test')
    test_loaders = [data.DataLoader(dataset=valid, batch_size=1, shuffle=False, num_workers=0) for valid in test_set]
    print(f"成功挂载 {len(test_loaders)} 个织物图像测试加载器。")

    # 4. 载入最佳权重参数 (强物理记忆热启动)
    print("初始化深层自适应 ADMM-Restormer 拓扑解耦模型...")
    model = net.denoise_Net_admm_restormer(opt)
    pretained_path = opt["pretained_path"]
    
    if not os.path.exists(pretained_path):
        print(f"❌ 错误：未找到指定的预训练权重文件 [{pretained_path}]，请检查路径。")
        exit(1)
        
    print(f"--> 正在无损注入历史最佳权重: {pretained_path}")
    state = torch.load(pretained_path, map_location=device)
    state_dict = state['state_dict'] if 'state_dict' in state else state
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 5. 评测容器初始化
    num_classes = opt.get("num_classes", 24)
    all_preds = []
    all_targets = []

    print("\n🚀 开始执行全量陌生织物样本评估（田字格无重叠 4 视点联合表决机制）...")
    
    total_inference_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders):
            for batch_data in test_loader:
                
                true_label = 0
                
                # 步骤 A：多层元组结构动态剥离，深度提取正宗的真标签
                if isinstance(batch_data, (tuple, list)):
                    if len(batch_data) == 3:
                        _, img_L, _ = batch_data
                        true_label = loader_idx // (len(test_loaders) // num_classes)
                    elif len(batch_data) == 2:
                        img_L, true_label_tensor = batch_data
                        true_label = true_label_tensor.item() if isinstance(true_label_tensor, torch.Tensor) else int(true_label_tensor)
                    else:
                        img_L = batch_data[0]
                        true_label = loader_idx // (len(test_loaders) // num_classes)
                else:
                    img_L = batch_data
                    true_label = loader_idx // (len(test_loaders) // num_classes)

                # 步骤 B: 强制转换为张量
                if not isinstance(img_L, torch.Tensor):
                    img_L = torch.from_numpy(img_L)

                # 步骤 C：多级维度质检防护线
                img_dim = img_L.dim()
                if img_dim == 2:
                    img_L = img_L.unsqueeze(0).unsqueeze(0)
                elif img_dim == 3:
                    img_L = img_L.unsqueeze(0)
                elif img_dim <= 1:
                    if isinstance(batch_data, (tuple, list)) and len(batch_data) > 0:
                        img_L = batch_data[0]
                        if img_L.dim() == 3: img_L = img_L.unsqueeze(0)
                        true_label_tensor = batch_data[1] if len(batch_data) > 1 else torch.tensor([0])
                        true_label = true_label_tensor.item() if isinstance(true_label_tensor, torch.Tensor) else int(true_label_tensor)
                    else:
                        continue

                # 步骤 D：占位噪声级与显存搬运
                noise_level = torch.zeros((img_L.size(0), 1, 1, 1)).to(device)
                img_L = img_L.to(device, non_blocking=True)
                
                B, C, H, W = img_L.shape
                patch_size = 128
                
                # 执行 128x128 田字格空间无重叠切块
                h_coords = [0, H - patch_size]
                w_coords = [0, W - patch_size]
                
                start_time = time.time()
                patch_outputs_list = []
                for hc in h_coords:
                    for wc in w_coords:
                        img_patch = img_L[:, :, hc:hc+patch_size, wc:wc+patch_size]
                        outputs_patch, _ = model(img_patch, noise_level)
                        patch_outputs_list.append(outputs_patch)
                
                # 4 视点联合表决
                avg_outputs = torch.mean(torch.stack(patch_outputs_list), dim=0)
                total_inference_time += (time.time() - start_time)
                
                all_preds.append(avg_outputs.cpu())
                all_targets.append(int(true_label))
                total_samples += 1

    # 6. 后处理与核心分类指标统计
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    # 计算全局 Top-1 和 Top-5 准确率
    top1_hit, top5_hit = calculate_topk_accuracy(preds_tensor, targets_tensor, topk=(1, 5))
    top1_acc = (top1_hit / total_samples) * 100
    top5_acc = (top5_hit / total_samples) * 100

    # 构建混淆矩阵
    _, preds_labels = torch.max(preds_tensor, 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(targets_tensor.numpy(), preds_labels.numpy()):
        if t < num_classes and p < num_classes:
            cm[t, p] += 1

    # 计算单类指标
    class_precision = []
    class_recall = []
    class_f1 = []

    t_classwise = PrettyTable(['Class ID', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        p = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        class_precision.append(p)
        class_recall.append(r)
        class_f1.append(f1)
        
        t_classwise.add_row([f'Class {i:02d}', f'{p:.2f}', f'{r:.2f}', f'{f1:.2f}'])

    macro_precision = np.mean(class_precision)
    macro_recall = np.mean(class_recall)
    macro_f1 = np.mean(class_f1)

    # 7. 优雅的三线表输出展示
    t_global = PrettyTable(['全局综合评测指标 (Global Metrics)', '学术考核得分 (Value)'])
    t_global.add_row(['全局单类准确率 Top-1 Accuracy', f'{top1_acc:.2f}%'])
    t_global.add_row(['全局综合准确率 Top-5 Accuracy', f'{top5_acc:.2f}%'])
    t_global.add_row(['宏平均精确率 Macro-Precision', f'{macro_precision:.2f}%'])
    t_global.add_row(['宏平均召回率 Macro-Recall', f'{macro_recall:.2f}%'])
    t_global.add_row(['平衡综合得分 Macro-F1 Score', f'{macro_f1:.2f}%'])
    t_global.add_row(['单张原图平均推理时耗 (Latency)', f'{(total_inference_time / total_samples):.4f}s'])

    logger.info(f"\n[1] 全局性能统计结果:\n{t_global}")
    logger.info(f"\n[2] 24类单类精细化统计报告:\n{t_classwise}")
    
    print(f"\n🏆 全量样本分类性能评测完成！")
    print("\n[1] 全局综合评测指标汇报表:")
    print(t_global)
    print("\n[2] 24类单类精细化性能拆解表:")
    print(t_classwise)

    # 8. 绘制 24x24 混淆矩阵
    plt.figure(figsize=(14, 12), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix of Multi-Class Textile Pattern Classification', fontsize=14, pad=20)
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f'C{i}' for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [f'C{i}' for i in range(num_classes)])
    
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=8)

    plt.ylabel('True Textile Class Label', fontsize=12)
    plt.xlabel('Predicted Textile Class Label', fontsize=12)
    plt.tight_layout()
    
    save_fig_path = os.path.join(opt['log_path'], 'Textile_Classification_Confusion_Matrix.png')
    plt.savefig(save_fig_path, bbox_inches='tight')
    print(f"\n📊 完美的 24x24 混淆矩阵热力图已成功绘制：{save_fig_path}")