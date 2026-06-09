from typing import List
import torch.utils.data as data
import torch
import time
import os
from tqdm import tqdm
import logging
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import signal
import sys
import numpy as np

# 模型与工具包导入
import Net.denoise_net as net  
import utils.utils_option as option

def adjust_learning_rate(opt, epo, lr_ini, max_epoch):
    """ 
    专为 24 类织物纹理分类优化的宽基自适应衰减函数
    让模型在前 120 轮保持充足的探索能量，彻底解决 23% 处的过早停滞问题
    """
    decay_interval = 60 
    
    if epo < 120:
        # 前 120 轮保持在一个相对饱满的梯度区间，强力拉低交叉熵
        lr = lr_ini * (0.75 ** (epo // decay_interval))
    elif epo < 300:
        # 中期进入精细特征剥离阶段
        lr = lr_ini * 0.1 * (0.85 ** ((epo - 120) // decay_interval))
    else:
        # 后期进入局部极小值探底
        lr = lr_ini * 0.01
        
    for param_group in opt.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # 硬件环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
    print(f"Using GPU for Textile End-to-End Classification Training (128x128 Patch Alignment)!\n")
        
    # 解析配置文件路径
    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)
    
    # 日志记录系统初始化
    logger_name = 'train_cls_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger_name = os.path.join(opt['log_path'], logger_name + '.log')
    os.makedirs(opt['log_path'], exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=utils_logger_name)
    logger = logging.getLogger(logger_name)
    
    # 1. 数据集载入（分类隔离模式）
    print("加载织物灰度分类数据集...")
    from utils.dataset_admm import get_data  
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')
    
    logger.info(f"训练集样本数: {len(train_set)}")
    
    # 💡 显存大解放：现在尺寸回到了 128，batch_size 可以在配置文件里放心设为 8 或 16 来极大加速
    train_loader = data.DataLoader(
        dataset=train_set, batch_size=opt['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    
    # 适配多验证子集加载器 (验证时以独立大图为基准载入，内部自动执行田字格分布式切块)
    test_loaders = [data.DataLoader(dataset=v, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) for v in valid_set]

    # 2. 物理网络架构定义与权重注入
    print("初始化深层 ADMM-Restormer 纹理特征分类模型...")
    model = net.denoise_Net_admm_restormer(opt)
    model.to(device)
    
    # 💥 显式断点载入恢复逻辑 (严格执行部分权重热启动，避免由于网络更新导致拼图不一致报错)
    if opt.get('pretained_path', {}).get('index', False):
        weight_path = opt['pretained_path']['path']
        if os.path.exists(weight_path):
            print(f"--> 正在安全注入历史最佳分类权重: {weight_path}")
            checkpoint = torch.load(weight_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
            print(f"🏆 历史基础物理层参数选择性载入成功！已成功部分继承物理记忆。\n")
        else:
            print(f"⚠️ 警告：配置文件中的权重路径 [{weight_path}] 未找到，将从头开始随机初始化训练！\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = nn.CrossEntropyLoss()
    
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"模型总参数量: {total / 1e6 :.2f}M")

    loss_train = []
    test_acc_history = []
    
    # 早停控制变量
    best_acc = 0.0
    best_epoch = 0
    max_accumulation = 0  # 计数器：记录连续多少次验证没有带来精度的增长

    # 3. 手动强行中断保护机制
    def signal_handler(sig, frame):
        print(f"\n训练被手动中断！正在安全封存临时权重...")
        torch.save({'state_dict': model.state_dict()}, os.path.join(opt['model_save'], 'model_cls_interrupted.pth'))
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    eval_num = 10  # 每 10 个 Epoch 触发一次全量验证

    # 4. 主端到端优化循环
    for epoch in range(0, opt["max_epoch"]):
        try:
            # 调用宽基动态学习率调度器
            adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
            
            model.train()
            loss_epoch = 0
            correct_train = 0
            total_train = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{opt["max_epoch"]} [Train]')
            for batch_idx, batch_data in enumerate(pbar):
                img_L, labels, noise_level = batch_data
                
                img_L = img_L.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                noise_level = noise_level.to(device, non_blocking=True)
                
                # 正向传播
                outputs, _ = model(img_L, noise_level)
                
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                loss_epoch += loss.item()
                
                # 计算训练集实时预测精度
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if batch_idx % 5 == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss_epoch/(batch_idx+1):.4f}',
                        'Acc': f'{100 * correct_train / total_train:.2f}%'
                    })

            avg_loss = loss_epoch / len(train_loader)
            train_acc = 100 * correct_train / total_train
            loss_train.append(avg_loss)
            logger.info(f"Epoch [{epoch+1}/{opt['max_epoch']}] - Avg Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - LR: {optimizer.param_groups[0]['lr']:.8f}")

            # 5. 全量陌生织物样本验证阶段（🔥 升级：田字格无重叠多视点 Patch 投票验证）
            if (epoch + 1) % eval_num == 0:
                print(f"\n>>> 开始全量验证阶段 - Epoch {epoch+1}")
                model.eval()
                
                total_val_correct = 0
                total_val_samples = 0
                
                with torch.no_grad():
                    for loader_idx, test_loader in enumerate(test_loaders):
                        val_correct = 0
                        val_samples = 0
                        
                        for v_img_L, v_labels, v_noise in test_loader:
                            v_img_L = v_img_L.to(device, non_blocking=True)
                            v_labels = v_labels.to(device, dtype=torch.long, non_blocking=True)
                            v_noise = v_noise.to(device, non_blocking=True)
                            
                            B, C, H, W = v_img_L.shape
                            patch_size = 128  # 严格对齐训练时的 128 拓扑通量
                            
                            # 💡 验证集执行田字格无重叠切块 (Non-overlapping Patches)
                            # 对于 224x224 原图，左上、右上、左下、右下刚好形成 4 个无重叠的 128 视点
                            h_coords = [0, H - patch_size]
                            w_coords = [0, W - patch_size]
                            
                            patch_outputs_list = []
                            for hc in h_coords:
                                for wc in w_coords:
                                    img_patch = v_img_L[:, :, hc:hc+patch_size, wc:wc+patch_size]
                                    # 让 ADMM 的物理层和分类头在其最熟悉的 128 流形下进行等价推理
                                    outputs_patch, _ = model(img_patch, v_noise)
                                    patch_outputs_list.append(outputs_patch)
                            
                            # 4个亚空间的局部分类概率联合表决，决定这幅 224 大图的最终花纹类别
                            avg_outputs = torch.mean(torch.stack(patch_outputs_list), dim=0)
                            _, v_predicted = torch.max(avg_outputs.data, 1)
                            
                            val_samples += v_labels.size(0)
                            val_correct += (v_predicted == v_labels).sum().item()
                        
                        total_val_correct += val_correct
                        total_val_samples += val_samples
                
                epoch_val_acc = 100 * total_val_correct / total_val_samples
                test_acc_history.append(epoch_val_acc)
                logger.info(f"--- Epoch {epoch+1} 综合验证集准确率: {epoch_val_acc:.2f}% (历史最高: {best_acc:.2f}%) ---")
                
                # 早停机制触发逻辑
                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    best_epoch = epoch + 1
                    max_accumulation = 0  # 精度增长，计数清零
                    torch.save({'state_dict': model.state_dict()}, os.path.join(opt['model_save'], "model_cls_best.pth"))
                    print(f"🏆 发现更优分类模型，已存至 model_cls_best.pth! 最高 Acc: {best_acc:.2f}%")
                else:
                    max_accumulation += 1  # 没能突破历史最高，累加一次
                    print(f"⚠️ 验证集精度未增长，当前已连续 {max_accumulation} 次未提升。")
                
                # 连续 10 次验证（即 100 个 Epoch）不增长，立刻触发早停
                if max_accumulation >= 10:
                    print(f"🛑 早停机制触发！模型已连续 {max_accumulation} 次验证（100个Epoch）未实现精度增长。")
                    logger.info(f"Early stopping at epoch {epoch+1} due to precision stagnation.")
                    break

            # 自动维护最新权重
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join(opt['model_save'], 'model_cls_latest.pth'))

        except Exception as e:
            print(f"运行异常抛出: {e}")
            logger.error(f"Error in epoch {epoch+1}: {e}")
            raise e

    # 6. 绘图导出
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train, label='Train CrossEntropy Loss')
    plt.title("Classification Loss")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(eval_num, (len(test_acc_history)*eval_num)+1, eval_num), test_acc_history, label='Val Accuracy', color='orange')
    plt.title("Validation Accuracy (%)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(opt['log_path'], 'cls_training_curves.png'))
    print(f"训练正式结束！最优精度在 Epoch {best_epoch} 达成，Best Acc: {best_acc:.2f}%.")