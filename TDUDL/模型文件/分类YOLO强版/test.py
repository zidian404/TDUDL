from typing import List
import os
import time
import random
from glob import glob

import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import logging
from prettytable import PrettyTable

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
    # 1. 硬件与随机种子
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    seed_ = 1234
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    # 2. 解析配置
    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)

    logger_name = 'test_cls_eval_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    print("Using GPU for Textile Classification Evaluation (YOLO-style 128x128)...\n")

    # 3. 加载 test 数据集（单 Dataset，和 YOLO 一样按子文件夹分类）
    print("加载 date-gray/test 织物测试集...")
    test_set = get_data(opt, 'test')          # dataset_admm_denose(opt['test'], 'test')
    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=opt.get('batch_size', 1),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"测试样本数: {len(test_set)}")

    # 4. 载入模型与最佳权重
    print("初始化 ADMM-Restormer 分类模型并载入最佳权重...")
    model = net.denoise_Net_admm_restormer(opt)
    pretained_path = opt["pretained_path"]

    if not os.path.exists(pretained_path):
        print(f"❌ 未找到预训练权重文件: {pretained_path}")
        exit(1)

    state = torch.load(pretained_path, map_location=device)
    state_dict = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 5. 评测容器
    num_classes = opt.get("num_classes", 24)
    all_preds = []
    all_targets = []

    total_inference_time = 0.0
    total_samples = 0

    print("\n🚀 开始 YOLO 风格单视角 128x128 全量测试评估...")
    with torch.no_grad():
        for batch_data in test_loader:
            img_L, labels, noise_level = batch_data

            img_L = img_L.to(device, non_blocking=True)          # [B,1,128,128]
            labels = labels.to(device, dtype=torch.long)
            noise_level = noise_level.to(device, non_blocking=True)

            start_time = time.time()
            outputs, _ = model(img_L, noise_level)               # [B, num_classes]
            total_inference_time += (time.time() - start_time)

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())
            total_samples += labels.size(0)

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # 6. Top-1 / Top-5
    top1_hit, top5_hit = calculate_topk_accuracy(preds_tensor, targets_tensor, topk=(1, 5))
    top1_acc = (top1_hit / total_samples) * 100
    top5_acc = (top5_hit / total_samples) * 100

    # 混淆矩阵
    _, preds_labels = torch.max(preds_tensor, 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(targets_tensor.numpy(), preds_labels.numpy()):
        if t < num_classes and p < num_classes:
            cm[t, p] += 1

    # 单类指标
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

    # 7. 全局指标表
    t_global = PrettyTable(['Global Metric', 'Value'])
    t_global.add_row(['Top-1 Accuracy', f'{top1_acc:.2f}%'])
    t_global.add_row(['Top-5 Accuracy', f'{top5_acc:.2f}%'])
    t_global.add_row(['Macro-Precision', f'{macro_precision:.2f}%'])
    t_global.add_row(['Macro-Recall', f'{macro_recall:.2f}%'])
    t_global.add_row(['Macro-F1', f'{macro_f1:.2f}%'])
    t_global.add_row(['Avg Latency / Image', f'{(total_inference_time / total_samples):.4f}s'])

    logger.info(f"\n[1] Global Metrics:\n{t_global}")
    logger.info(f"\n[2] Class-wise Metrics:\n{t_classwise}")

    print("\n🏆 全量 test 样本分类评测完成！")
    print("\n[1] Global Metrics:")
    print(t_global)
    print("\n[2] 24-class Metrics:")
    print(t_classwise)

    # 8. 混淆矩阵可视化
    plt.figure(figsize=(14, 12), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix of Multi-Class Textile Classification', fontsize=14, pad=20)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f'C{i}' for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [f'C{i}' for i in range(num_classes)])

    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                plt.text(
                    j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8
                )

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()

    save_fig_path = os.path.join(opt['log_path'], 'Textile_Classification_Confusion_Matrix.png')
    plt.savefig(save_fig_path, bbox_inches='tight')
    print(f"\n📊 24x24 混淆矩阵热力图已保存：{save_fig_path}")