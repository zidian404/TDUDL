from typing import List, Dict
import torch.utils.data as data
import torch
import time
import os
from tqdm import tqdm
import logging
from torch import optim
import matplotlib.pyplot as plt
from math import log
import signal
import sys
import numpy as np
from glob import glob

# 🔥 所有导入
import Net.denoise_net as net
from utils.dataset_admm import get_data
from utils.loss_function import loss_function
import utils.utils_option as option
import utils.utils_image as image
from utils import utils_logger
from utils.global_id_map import load_global_id_map, generate_global_id_map  # 🔥 全局id映射

def adjust_learning_rate(opt, epo, lr_ini, max_epoch):
    P1 = 50
    P2 = 200 - P1
    if epo < P1:
        lr = lr_ini * (0.65 ** (epo // (P1 // log(0.1, 0.65))))
    else:
        lr = lr_ini * 0.1 * (0.85 ** ((epo - P1) // (P2 // log(0.1, 0.85))))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # 基本设置
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
    print(f"Using GPU for training!\n")
    
    # 🔥 修复1：先解析opt
    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)
    
    # 🔥 修复2：安全生成全局id映射
    try:
        path2id, total_n_samples = load_global_id_map()
        print(f"✅ 加载全局id映射: {total_n_samples} 张图")
    except FileNotFoundError:
        print("⚠️  id_map.json 不存在，正在生成...")
        path2id, total_n_samples = generate_global_id_map(json_path)
        print(f"📊 Di_param 大小设为: {total_n_samples}")
    
    # 日志
    logger_name = 'train' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # 数据集 - 使用全局唯一id
    print("加载数据集...")
    train_set = get_data(opt, 'train', path2id=path2id)
    valid_set = get_data(opt, 'valid', path2id=path2id)
    
    logger.info(f"训练集大小: {len(train_set)}")
    logger.info(f"验证集数量: {len(valid_set)} (每个 sigma 一个 Dataset)")
    
    # 数据加载配置
    train_loader = data.DataLoader(
        dataset=train_set, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    test_loaders: List[data.DataLoader] = []
    for valid in valid_set:
        test_loaders.append(data.DataLoader(
            dataset=valid, 
            batch_size=1,
            shuffle=False, 
            num_workers=0, 
            drop_last=False, 
            pin_memory=True
        ))

    # 模型初始化 - 使用全局图数
    print("初始化模型...")
    model = net.denoise_Net_admm_restormer(opt, n_samples=total_n_samples)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = loss_function(opt['loss_function_index'])
    
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Number of parameter: {total / 1e6 :.2f}M")
    logger.info("start training...")

    # 训练记录变量
    start = time.time()
    loss_train = []
    test__loss = []
    test__psnr = []
    test__ssim = []
    best_psnr = 0
    best_epoch = 0
    batch_accumulation = 1
    max_accumulation = 0
    psnr_val_rgb = 0

    if opt["pretained_path"]["index"]:
        state = torch.load(opt['pretained_path']["path"], weights_only=False)
        model.load_state_dict(state['state_dict'], strict=False)

    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.85, patience=5,
        threshold=1e-3, threshold_mode='abs', min_lr=0, eps=1e-8
    )

    eval_num = 5

    # 中断处理
    def signal_handler(sig, frame):
        print(f"\n训练被中断！正在保存当前进度...")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, os.path.join(opt['model_save'], 'model_interrupted.pth'))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 主训练循环
    for epoch in range(0, opt["max_epoch"]):
        try:
            if epoch < 200:
                adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
            else:
                reduce_schedule.step(psnr_val_rgb)

            model.train()
            loss_epoch = 0
            batch_count = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
            
            for batch_idx, batch_data in enumerate(pbar):
                img_H, img_L, noise_level, ids = batch_data
                
                img_H = img_H.to(device, non_blocking=True)
                img_L = img_L.to(device, non_blocking=True)
                noise_level = noise_level.to(device, non_blocking=True)
                ids = ids.to(device, non_blocking=True)
                
                output, preds = model(img_L, noise_level, ids)

                if isinstance(output, (list, tuple)):
                    output = output[0]
                if output.dim() == 3:
                    output = output.unsqueeze(1)
                if img_H.dim() == 3:
                    img_H = img_H.unsqueeze(1)

                loss = criterion(output, img_H) / batch_accumulation
                loss.backward()
                
                if (batch_idx + 1) % batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                loss_epoch += loss.item() * batch_accumulation
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({'Loss': f'{loss_epoch/batch_count:.4f}'})

            avg_loss = loss_epoch / len(train_loader)
            loss_train.append(avg_loss)
            logger.info(f"epoch:[{epoch + 1}/{opt['max_epoch']}], 平均loss: {avg_loss:.4f}")

            # 验证阶段
            if (epoch + 1) % eval_num == 0:
                model.eval()
                test_loss, test_psnr, test_ssim, valid_batches = 0, 0, 0, 0
                
                val_pbar = tqdm(total=sum(len(tl) for tl in test_loaders), desc=f'Epoch {epoch+1} [Val]')
                
                with torch.no_grad():
                    for test_loader in test_loaders:
                        for batch_idx, batch_data in enumerate(test_loader):
                            v_img_H, v_img_L, v_noise, v_ids = batch_data

                            v_img_H = v_img_H.to(device)
                            v_img_L = v_img_L.to(device)
                            v_noise = v_noise.to(device)
                            v_ids = v_ids.to(device)

                            v_out, _ = model(v_img_L, v_noise, v_ids)
                            if isinstance(v_out, (list, tuple)):
                                v_out = v_out[0]
                            if v_out.dim() == 3:
                                v_out = v_out.unsqueeze(1)
                            if v_img_H.dim() == 3:
                                v_img_H = v_img_H.unsqueeze(1)
                            
                            if v_out.shape[2:] != v_img_H.shape[2:]:
                                v_out = torch.nn.functional.interpolate(
                                    v_out, size=v_img_H.shape[2:], 
                                    mode='bilinear', align_corners=False
                                )

                            test_loss += criterion(v_out, v_img_H).item()
                            v_out_u = image.tensor2uint(v_out)
                            v_img_H_u = image.tensor2uint(v_img_H)
                            
                            current_psnr = image.calculate_psnr(v_out_u, v_img_H_u)
                            test_psnr += current_psnr
                            test_ssim += image.calculate_ssim(v_out_u, v_img_H_u)
                            valid_batches += 1
                            
                            val_pbar.set_postfix({'curr_PSNR': f'{current_psnr:.2f}'})
                            val_pbar.update(1)
                
                val_pbar.close()

                if valid_batches > 0:
                    psnr_val_rgb = test_psnr / valid_batches
                    ssim_val_rgb = test_ssim / valid_batches
                    avg_test_loss = test_loss / valid_batches
                    
                    print(f"\n[Validation Result] PSNR: {psnr_val_rgb:.4f} | SSIM: {ssim_val_rgb:.4f} | Loss: {avg_test_loss:.4f}")
                    
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch + 1
                        torch.save(
                            {'state_dict': model.state_dict()},
                            os.path.join(opt['model_save'], "model_best.pth")
                        )
                        print(f"*** Best Model Saved! Best PSNR: {best_psnr:.4f} ***")
                        max_accumulation = 0
                    else:
                        max_accumulation += 1
                        print(f"--- No improvement for {max_accumulation} epoch(s). Best: {best_psnr:.4f} ---")

                    logger.info(f'[epoch {epoch+1} PSNR: {psnr_val_rgb:.4f} Best: {best_psnr:.4f}]')
                    test__loss.append(avg_test_loss)
                    test__psnr.append(psnr_val_rgb)
                    test__ssim.append(ssim_val_rgb)

            # 保存最新模型
            torch.save(
                {'epoch': epoch+1, 'state_dict': model.state_dict()},
                os.path.join(opt['model_save'], 'model_latest.pth')
            )
            
            # 早停逻辑
            if max_accumulation >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

        except Exception as e:
            logger.error(f"训练出错: {e}")
            torch.save(
                {'state_dict': model.state_dict()},
                os.path.join(opt['model_save'], 'model_error.pth')
            )
            raise e

    # 训练完成绘图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_train, label='Train Loss')
    plt.savefig(os.path.join(opt['log_path'], 'loss_curve.png'))
    print("训练结束，曲线已保存。")
