import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any
from Net.restormer_arch import Restormer11

##########################################################################
# Basic modules

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=(stride, stride)
    )

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=(stride, stride), bias=bias
    )

def conv_down(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.Conv2d(
        in_chn, out_chn, kernel_size,
        stride=(stride, stride),
        padding=(kernel_size - 1) // 2,
        bias=bias
    )

def conv_up(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.ConvTranspose2d(
        in_chn, out_chn, kernel_size,
        stride=(stride, stride),
        padding=(kernel_size - 1) // 2,
        output_padding=stride - 1,
        bias=bias
    )

##########################################################################
# HyPaNet

class HyPaNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 1,
        nc: int = 64,
        out_nc: int = 5,   # 输出 5 个超参数
    ):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus()
        )

    def forward(self, x: Tensor):
        # x: [B,1,1,1]
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6        # [B,5,1,1]
        return x

##########################################################################
# HeadNet

class HeadNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_size: int):
        super(HeadNet, self).__init__()

        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size,
                      padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, y: Any, sigma):
        # sigma: [B,1,1,1] -> [B,1,H,W]
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        return x

##########################################################################
# Di 卷积及其"转置"卷积（按 batch） - 修改版，支持维度自动处理

def apply_Di(X: torch.Tensor, D_i: torch.Tensor) -> torch.Tensor:
    """
    X:   [B, Cx, H, W]
    D_i: [B, Cy, Cx, k, k] 或自动处理额外维度
    返回 Y: [B, Cy, H, W]
    """
    B, Cx, H, W = X.shape
    
    # 自动处理维度：squeeze掉末尾多余维度
    while D_i.dim() > 5:
        D_i = D_i.squeeze(-1)
    if D_i.dim() != 5:
        raise ValueError(f"Di dim must be 5 after squeezing, got {D_i.dim()}")
    
    _, Cy, Cx_d, k, _ = D_i.shape
    assert Cx_d == Cx, f"Di Cx={Cx_d} must match X Cx={Cx}"
    assert D_i.size(0) == B, f"Di batch size {D_i.size(0)} must match X batch size {B}"

    # unfold X -> [B, Cx*k*k, N]
    patches = F.unfold(X, kernel_size=k, padding=k // 2)   # [B, Cx*k*k, N]
    B_u, CK, N = patches.shape
    assert B_u == B

    patches = patches.transpose(1, 2)                      # [B, N, Cx*k*k]

    # 展平卷积核
    K_flat = D_i.view(B, Cy, Cx * k * k)                   # [B, Cy, Cx*k*k]

    # [B, Cy, N] = [B, Cy, Cx*k*k] @ [B, Cx*k*k, N]
    Y_flat = torch.bmm(K_flat, patches.transpose(1, 2))    # [B, Cy, N]

    # 折回 [B, Cy, H, W]
    Y = F.fold(Y_flat, output_size=(H, W), kernel_size=1)
    return Y

def apply_Di_T(Y: torch.Tensor, D_i: torch.Tensor) -> torch.Tensor:
    """
    Y:   [B, Cy, H, W]
    D_i: [B, Cy, Cx, k, k] 或自动处理额外维度
    返回 X_grad: [B, Cx, H, W]
    """
    B, Cy, H, W = Y.shape
    
    # 自动处理维度：squeeze掉末尾多余维度
    while D_i.dim() > 5:
        D_i = D_i.squeeze(-1)
    if D_i.dim() != 5:
        raise ValueError(f"Di dim must be 5 after squeezing, got {D_i.dim()}")
    
    _, Cy_d, Cx, k, _ = D_i.shape
    assert Cy_d == Cy, f"Di Cy={Cy_d} must match Y Cy={Cy}"
    assert D_i.size(0) == B, f"Di batch size {D_i.size(0)} must match Y batch size {B}"

    # unfold Y -> [B, Cy*k*k, N]
    patches = F.unfold(Y, kernel_size=k, padding=k // 2)   # [B, Cy*k*k, N]
    B_u, CK, N = patches.shape
    assert B_u == B

    patches = patches.transpose(1, 2)                      # [B, N, Cy*k*k]

    # D_i^T: [B, Cx, Cy, k, k]
    K_T = D_i.permute(0, 2, 1, 3, 4).contiguous()         # [B, Cx, Cy, k, k]
    K_T_flat = K_T.view(B, Cx, Cy * k * k)                # [B, Cx, Cy*k*k]

    # [B, Cx, N] = [B, Cx, Cy*k*k] @ [B, Cy*k*k, N]
    X_flat = torch.bmm(K_T_flat, patches.transpose(1, 2))  # [B, Cx, N]

    X_grad = F.fold(X_flat, output_size=(H, W), kernel_size=1)
    return X_grad

##########################################################################
# BodyNet：用共享 S + 每样本 Di，Z / beta 更新形式保持原样

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T, Di_param):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S = S
        self.S_T = S_T
        self.Di_param = Di_param

    def forward(self, ids, X_in, Y, Z, beta, alpha, rho,
                gamma, samfeats, enc, dec):

        device = X_in.device
        X = X_in
        ids = ids.to(torch.long).to(device)

        # 取本 batch 对应的 Di: [B,Cy,Cx,k,k] - 自动维度处理
        Di_batch = self.Di_param[ids]

        # ---- X-step: 用 S + Di ----
        Y_S = self.S(X)                 # [B,Cy,H,W]
        Y_D = apply_Di(X, Di_batch)     # [B,Cy,H,W]
        Y_hat = Y_S + Y_D

        res = Y_hat - Y                 # [B,Cy,H,W]
        grad_S = self.S_T(res)          # [B,Cx,H,W]
        grad_D = apply_Di_T(res, Di_batch)
        grad = grad_S + grad_D          # [B,Cx,H,W]

        # alpha, rho: [B,1,1,1]，自动广播
        X_term = X - Z + beta
        X_out = X - alpha * (grad + rho * X_term)

        # Z-step（Restormer）
        rho_ = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_], dim=1),
            samfeats, enc, dec, stage_inter=True
        )

        # beta-step
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z

        return X_out, Z, beta, samfeats, enc_, dec_

##########################################################################
# 主网络：共享 S + 每样本 Di

class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt, n_samples: int = 1600):
        super(denoise_Net_admm_restormer, self).__init__()

        self.n_channels = opt["n_channels"]   # 图像域通道 Cy
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        # 系数域通道 Cx
        self.m_channels = 16
        self.stride = 1

        # Restormer (Z-step): 输入 m_channels+1 -> 输出 m_channels
        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,   # 16+1=17
            out_channels=self.m_channels,       # 16
            dim=self.m_channels
        )

        # ---- 共享字典 S / S_T + 每样本 Di 参数 ----
        k = self.d_size
        Cx = self.m_channels       # 系数域
        Cy = self.n_channels       # 图像域

        # S: 系数域 X[Cx] -> 图像域 Y[Cy]
        self.S = nn.Conv2d(Cx, Cy, k, padding=k // 2, bias=True)
        # S_T: 图像域 Y[Cy] -> 系数域 X[Cx]
        self.S_T = nn.Conv2d(Cy, Cx, k, padding=k // 2, bias=True)

        # 每样本 Di: [N, Cy, Cx, k, k]
        self.Di_param = nn.Parameter(
            0.01 * torch.randn(n_samples, Cy, Cx, k, k)
        )

        self.body = BodyNet(self.unet, self.S, self.S_T, self.Di_param)

        # 每个 stage 的超参数网络
        self.hypa_list_: nn.ModuleList = nn.ModuleList()
        for _ in range(self.stage):
            self.hypa_list_.append(HyPaNet(in_nc=1, out_nc=5))

    def forward(self, input, sigma, ids):
        """
        input: [B,Cy,H,W] 噪声图像 Y
        sigma: [B] / [B,1] / [B,1,1] / [B,1,1,1]
        ids:   [B] long, 样本索引 0~n_samples-1
        """
        device = input.device
        ids = ids.to(torch.long).to(device)

        # sigma -> [B,1,1,1]
        sigma = sigma.to(device)
        sigma = sigma.view(sigma.size(0), 1, 1, 1)

        # 初始化 X^0：图像域 -> 系数域
        X_img0 = self.headnet(input, sigma)        # [B,Cy,H,W]
        X = self.S_T(X_img0)                       # [B,Cx,H,W]

        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)

        samfeats = enc = dec = None

        for k in range(self.stage):
            # HyPaNet: [B,5,1,1]
            hypas = self.hypa_list_[k](sigma)
            alpha = hypas[:, 0:1, :, :]   # [B,1,1,1]
            rho   = hypas[:, 1:2, :, :]
            gamma1 = hypas[:, 2:3, :, :]
            gamma2 = hypas[:, 3:4, :, :]
            gamma3 = hypas[:, 4:5, :, :]
            gamma  = [gamma1, gamma2, gamma3]

            # 取本 batch 对应的 Di: 自动维度处理
            Di_batch = self.Di_param[ids]

            if k == 0:
                # -------- k==0：按 ADMM 初始化逻辑 + 用 S+Di --------
                # X1 = (S+Di)(X)
                Y_S = self.S(X)                     # [B,Cy,H,W]
                Y_D = apply_Di(X, Di_batch)         # [B,Cy,H,W]
                X1  = Y_S + Y_D                     # 图像域

                # temp_back = S_T(X1) + Di^T(input)
                temp_back = self.S_T(X1) + apply_Di_T(input, Di_batch)  # [B,Cx,H,W]
                temp = temp_back - self.S_T(input)

                # X2 = (S+Di)(temp)
                Y_S_temp = self.S(temp)
                Y_D_temp = apply_Di(temp, Di_batch)
                X2 = Y_S_temp + Y_D_temp            # 图像域

                # 转回系数域
                X1_coef = self.S_T(X1)              # [B,Cx,H,W]
                X2_coef = self.S_T(X2)

                X_ = X2_coef + rho * X1_coef
                X = X1_coef - alpha * X_

                # Z-step
                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_map], dim=1),
                    stage_inter=True
                )
                # beta-step
                beta = gamma[1] * X - gamma[2] * Z

                # 中间输出：用 S+Di 重构图像
                Y_S_out = self.S(X)
                Y_D_out = apply_Di(X, Di_batch)
                output = Y_S_out + Y_D_out
                preds.append(output)

            else:
                # 其余阶段用 BodyNet（内部已用 S+Di）
                X, Z, beta, samfeats, enc, dec = self.body(
                    ids, X, input, Z, beta, alpha, rho,
                    gamma, samfeats, enc, dec
                )
                Y_S_out = self.S(X)
                Y_D_out = apply_Di(X, Di_batch)
                output = Y_S_out + Y_D_out
                preds.append(output)

        # -------- FINAL STEP：用 S + Di 再做一次 X 更新 + 重构 --------
        Di_batch = self.Di_param[ids]

        # temp = (S+Di)(X) - input
        Y_S_final = self.S(X)
        Y_D_final = apply_Di(X, Di_batch)
        temp = (Y_S_final + Y_D_final) - input         # [B,Cy,H,W]

        # X_1 = S_T(temp) + Di^T(temp)
        X_1 = self.S_T(temp) + apply_Di_T(temp, Di_batch)

        # X_2 = rho * (X - Z - beta)
        X_2 = rho * (X - Z - beta)
        X_out = X - alpha * (X_1 + X_2)

        # 最终重构
        Y_S_out = self.S(X_out)
        Y_D_out = apply_Di(X_out, Di_batch)
        output = Y_S_out + Y_D_out
        preds.append(output)

        return output, preds

##########################################################################
class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()

    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in
