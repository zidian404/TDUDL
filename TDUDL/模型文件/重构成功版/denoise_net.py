import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple

# 保持从您的目录正确导入
from Net.restormer_arch import Restormer11 

##########################################################################
# 1. 基础卷积工具函数 (Basic Modules)
##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=(stride, stride))

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=(stride, stride), bias=bias)

def conv_down(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias)

def conv_up(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.ConvTranspose2d(in_chn, out_chn, kernel_size, stride=stride, 
                              padding=(kernel_size - 1) // 2, output_padding=stride-1, bias=bias)

class ST(nn.Module):
    """ 软阈值算子 (Shrinkage-Thresholding) """
    def __init__(self):
        super(ST, self).__init__()
    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in

##########################################################################
# 2. 矩阵运算算子 (针对 Batch 维度的 Di 卷积)
##########################################################################

def apply_Di(X, D_i):
    """ X: [B, Cx, H, W], D_i: [B, Cy, Cx, k, k] -> [B, Cy, H, W] """
    B, Cx, H, W = X.shape
    _, Cy, _, k, _ = D_i.shape
    patches = F.unfold(X, kernel_size=k, padding=k // 2)
    patches = patches.transpose(1, 2)
    K_flat = D_i.view(B, Cy, Cx * k * k)
    Y_flat = torch.bmm(K_flat, patches.transpose(1, 2))
    return F.fold(Y_flat, output_size=(H, W), kernel_size=1)

def apply_Di_T(Y, D_i):
    """ Y: [B, Cy, H, W], D_i: [B, Cy, Cx, k, k] -> [B, Cx, H, W] """
    B, Cy, H, W = Y.shape
    _, _, Cx, k, _ = D_i.shape
    patches = F.unfold(Y, kernel_size=k, padding=k // 2)
    patches = patches.transpose(1, 2)
    K_T = D_i.permute(0, 2, 1, 3, 4).contiguous()  # 经典索引4，无越界风险
    K_T_flat = K_T.view(B, Cx, Cy * k * k)
    X_flat = torch.bmm(K_T_flat, patches.transpose(1, 2))
    return F.fold(X_flat, output_size=(H, W), kernel_size=1)

##########################################################################
# 3. 子网络定义 (Head, HyPa, Generator)
##########################################################################

class HeadNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False))

    def forward(self, y, sigma):
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        return self.head_x(torch.cat([y, sigma], dim=1))

class HyPaNet(nn.Module):
    def __init__(self, in_nc: int = 1, nc: int = 64, out_nc: int = 5):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x: Tensor):
        x = (x - 0.098) / 0.0566 
        return self.mlp(x) + 1e-6

class DiGenerator(nn.Module):
    """ 针对织物周期性图像设计的特异性字典生成器 """
    def __init__(self, in_channels, out_channels, k_size, m_channels):
        super(DiGenerator, self).__init__()
        self.out_c, self.in_c, self.k = out_channels, m_channels, k_size
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid())
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_channels * m_channels * k_size * k_size))

    def forward(self, y):
        b = y.size(0)
        feat = self.extractor(y)
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        attn = self.sa(torch.cat([avg_out, max_out], dim=1))
        pooled = self.global_pool(feat * attn).view(b, -1)
        return self.mlp(pooled).view(b, self.out_c, self.in_c, self.k, self.k)

##########################################################################
# 4. BodyNet 与主模型
##########################################################################

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T, in_conv, out_conv):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S, self.S_T = S, S_T
        self.in_conv = in_conv
        self.out_conv = out_conv

    def forward(self, X_in, Y, Z, beta, alpha, rho, gamma, Di_batch):
        # 1. X-step
        #res = (self.S(X_in) + apply_Di(X_in, Di_batch)) - Y
        #grad = self.S_T(res) + apply_Di_T(res, Di_batch)
        I_recon = self.S(X_in) + apply_Di(X_in, Di_batch)
        res_latent = self.S_T(I_recon) + apply_Di_T(I_recon, Di_batch)
        grad_latent = res_latent - X_in
        X_out = X_in - alpha * (grad_latent + rho * (X_in - Z + beta))
        
        # 2. Z-step (物理意义彻底隔离与解耦)
        rho_map = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        
        # 💥 核心净化：通过一层不带 Bias 的 1x1 卷积将带有强烈恒等映射痕迹的系数 X_out 投影到独立的特征空间
        # 这样即使 Restormer 内部强行进行 inp_img[:, :-1, :, :] 切片或跳连，也只能拿到经非线性/线性变换后的特征映射，无法无损搬运原始像素
        unet_input = self.in_conv(X_out)
        unet_input = torch.cat([unet_input, rho_map], dim=1) 
        
        Z_feat, _, _, _ = self.unet(unet_input, samfeats=None, enc_in=None, dec_in=None, stage_inter=False)
        
        # 经过反向投影，重构回我们真正的稀疏系数空间维度
        Z = self.out_conv(Z_feat)
        
        # 3. beta-step
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        return X_out, Z, beta

class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer, self).__init__()
        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]
        self.m_channels = 16

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)
        # 保持原本的 Restormer 初始化完全不变
        self.unet = Restormer11(inp_channels=self.m_channels+1, out_channels=self.m_channels, dim=self.m_channels)
        
        # 💥 新增：最简轻量化无后门投影层（1x1 卷积，参数量微乎其微，不破物理先验，只为拦截快捷后门）
        self.in_conv = nn.Conv2d(self.m_channels, self.m_channels, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(self.m_channels, self.m_channels, kernel_size=1, bias=False)
        
        # 共享字典 S 及其转置 S_T
        self.S = default_conv(self.m_channels, self.n_channels, self.d_size)
        self.S_T = default_conv(self.n_channels, self.m_channels, self.d_size)
        
        # 动态字典生成网络
        self.di_gen = DiGenerator(self.n_channels, self.n_channels, self.d_size, self.m_channels)
        
        # 将投影映射传递给 Body 块
        self.body = BodyNet(self.unet, self.S, self.S_T, self.in_conv, self.out_conv)
        self.hypa_list = nn.ModuleList([HyPaNet(in_nc=1, out_nc=5) for _ in range(self.stage)])

    def forward(self, input, sigma):
        # 1. 动态生成特异性字典
        Di_batch = self.di_gen(input) 

        # 2. 初始化
        sigma = sigma.view(sigma.size(0), 1, 1, 1).to(input.device)
        X = self.S_T(self.headnet(input, sigma)) # 系数域初始化

        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)

        # 3. ADMM 迭代阶段
        for k in range(self.stage):
            hypas = self.hypa_list[k](sigma)
            alpha, rho = hypas[:, 0:1], hypas[:, 1:2]
            gamma = [hypas[:, 2:3], hypas[:, 3:4], hypas[:, 4:5]]

            if k == 0:
                # 初始步：(S+Di) 组合保真项
                
                X1_img = self.S(X) + apply_Di(X, Di_batch)
                res_latent = self.S_T(X1_img) + apply_Di_T(X1_img, Di_batch)
                X_ = res_latent - X + rho * X
                X = X - alpha * X_
                rho_m = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                
                # 💥 初始步同步实施通道重定向
                unet_input = self.in_conv(X)
                unet_input = torch.cat([unet_input, rho_m], dim=1)
                
                Z_feat, _, _, _ = self.unet(unet_input, samfeats=None, enc_in=None, dec_in=None, stage_inter=False)
                Z = self.out_conv(Z_feat)
                
                beta = gamma[1] * X - gamma[2] * Z
            else:
                X, Z, beta = self.body(X,input, Z, beta, alpha, rho, gamma, Di_batch)
            
            # 记录中间重构结果（100% 由 $S(X) + Di(X)$ 组合产生，彻底断开像素后门）
            preds.append(self.S(X) + apply_Di(X, Di_batch))

        # 4. Final Step (最后一层 X 更新)
        #res_f = (self.S(X) + apply_Di(X, Di_batch)) - input
        #grad_f = self.S_T(res_f) + apply_Di_T(res_f, Di_batch)
        #X_out = X - alpha * (grad_f + rho * (X - Z - beta))
        
        final_out = self.S(X) + apply_Di(X, Di_batch)
        preds.append(final_out)

        return final_out, preds