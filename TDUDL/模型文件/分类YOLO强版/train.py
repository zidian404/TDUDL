import os
import sys
import time
import signal
import logging

import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm

import Net.denoise_net as net
import utils.utils_option as option


def adjust_learning_rate(optimizer, epoch, lr_ini):
    """
    保留你风格的分段衰减，但先简化一点以保证稳定：
    0-119: lr_ini
    120-299: lr_ini * 0.1
    300+: lr_ini * 0.01
    """
    if epoch < 120:
        lr = lr_ini
    elif epoch < 300:
        lr = lr_ini * 0.1
    else:
        lr = lr_ini * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 读取配置
    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)

    os.makedirs(opt['log_path'], exist_ok=True)
    os.makedirs(opt['model_save'], exist_ok=True)

    # 日志
    logger_name = 'train_cls_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    log_file = os.path.join(opt['log_path'], logger_name + '.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file
    )
    logger = logging.getLogger(logger_name)

    # ====================== 数据集 ======================
    print("加载训练 / 验证数据集...")
    from utils.dataset_admm import get_data
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = data.DataLoader(
        dataset=valid_set,
        batch_size=opt.get('val_batch_size', opt['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"训练集样本数: {len(train_set)}")
    logger.info(f"验证集样本数: {len(valid_set)}")

    # ====================== 模型 ======================
    print("初始化 ADMM-Restormer 分类模型...")
    model = net.denoise_Net_admm_restormer(opt).to(device)

    # 可选预训练
    if opt.get('pretained_path', {}).get('index', False):
        weight_path = opt['pretained_path']['path']
        if os.path.exists(weight_path):
            print(f"加载预训练权重: {weight_path}")
            checkpoint = torch.load(weight_path, map_location=device)
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            print("权重加载完成。\n")
        else:
            print(f"预训练权重不存在，忽略: {weight_path}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = nn.CrossEntropyLoss()

    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"模型总参数量: {total / 1e6 :.2f}M")

    train_loss_history = []
    val_acc_history = []

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    eval_num = 10
    early_stop_patience = 10

    # CTRL+C 保护
    def signal_handler(sig, frame):
        print("\n训练被手动中断，正在保存中断权重...")
        torch.save({'state_dict': model.state_dict()},
                   os.path.join(opt['model_save'], 'model_cls_interrupted.pth'))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    max_epoch = opt['max_epoch']

    # ====================== 训练主循环 ======================
    for epoch in range(max_epoch):
        adjust_learning_rate(optimizer, epoch, opt['lr'])

        model.train()
        loss_epoch = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{max_epoch} [Train]')
        for batch_idx, batch_data in enumerate(pbar):
            img_L, labels, noise_level = batch_data

            img_L = img_L.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            noise_level = noise_level.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs, _ = model(img_L, noise_level)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_epoch += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss_epoch / (batch_idx + 1):.4f}',
                    'Acc': f'{100 * correct_train / total_train:.2f}%'
                })

        avg_loss = loss_epoch / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        train_loss_history.append(avg_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{max_epoch}] "
            f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"LR: {optimizer.param_groups[0]['lr']:.6e}"
        )

        # ============ 每 eval_num 轮做一次验证 ============
        if (epoch + 1) % eval_num == 0:
            model.eval()
            total_val_correct = 0
            total_val_samples = 0

            with torch.no_grad():
                for v_img_L, v_labels, v_noise in valid_loader:
                    v_img_L = v_img_L.to(device, non_blocking=True)
                    v_labels = v_labels.to(device, dtype=torch.long, non_blocking=True)
                    v_noise = v_noise.to(device, non_blocking=True)

                    outputs, _ = model(v_img_L, v_noise)
                    _, predicted = torch.max(outputs.data, 1)

                    total_val_samples += v_labels.size(0)
                    total_val_correct += (predicted == v_labels).sum().item()

            val_acc = 100.0 * total_val_correct / total_val_samples
            val_acc_history.append(val_acc)

            logger.info(f"Epoch [{epoch + 1}] Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")
            print(f"Epoch [{epoch + 1}] Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({'state_dict': model.state_dict()},
                           os.path.join(opt['model_save'], 'model_cls_best.pth'))
                print(f"🏆 保存最佳模型，Val Acc = {best_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"验证集未提升，耐心计数: {patience_counter}/{early_stop_patience}")

            torch.save(
                {'epoch': epoch + 1, 'state_dict': model.state_dict()},
                os.path.join(opt['model_save'], 'model_cls_latest.pth')
            )

            torch.cuda.empty_cache()

            if patience_counter >= early_stop_patience:
                print("🛑 触发早停：验证集已连续 10 次未提升。")
                break

    # ====================== 画曲线 ======================
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history)
    plt.title("Train Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x_eval = list(range(eval_num, eval_num * len(val_acc_history) + 1, eval_num))
    plt.plot(x_eval, val_acc_history, color='orange')
    plt.title("Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(opt['log_path'], 'cls_training_curves.png'))

    print(f"训练结束，最佳验证精度出现在 Epoch {best_epoch}，Best Acc = {best_acc:.2f}%")