import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from typing import List
from einops import rearrange

##########################################################################
# Basic conv

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride
    )

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##########################################################################
# LayerNorm

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
# FNet

class FNet(nn.Module):
    def __init__(self, h_channel):
        super(FNet, self).__init__()
        self.conv1 = nn.Conv2d(h_channel, h_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(h_channel, h_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(h_channel, h_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(h_channel, h_channel, 3, 1, 1)

    def forward(self, f, enc, dec):
        skip_ = F.leaky_relu(self.conv1(enc) + self.conv2(dec), 0.1, inplace=True)
        out = f * torch.sigmoid(self.conv3(skip_)) + self.conv4(skip_) + f
        return out

##########################################################################
# CAB + CALayer

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias),
        ]
        self.body = nn.Sequential(*modules_body)
        self.CA = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
# SAM：在 n_feat 通道特征域工作（这里 n_feat = out_channels = 16）

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        # x: [B, n_feat, H, W], x_img: [B, n_feat, H, W]
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img

##########################################################################
# mergeblock

class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def forward(self, x, bridge):
        # x, bridge: [B, n_feat, H, W]
        out = torch.cat([x, bridge], 1)             # [B,2*n_feat,H,W]
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)                      # [B,num_subspace,H,W]
        V_t = sub.view(b_, self.num_subspace, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_ * w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out + x

##########################################################################
# FeedForward & Attention & TransformerBlock

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            3, 1, 1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            3, 1, 1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out, 'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim=16, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

##########################################################################
# PatchEmbed / Downsample / Upsample

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
# Restormer11

class Restormer11(nn.Module):
    def __init__(
        self,
        inp_channels=17,      # m_channels + 1 (16 + 1)
        out_channels=16,      # m_channels
        dim=16,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ):
        super(Restormer11, self).__init__()

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.CAB = CAB(dim, 3, reduction=4, bias=False, act=nn.PReLU())

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.f1_2 = FNet(dim)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.f2_3 = FNet(dim * 2)

        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.f3_4 = FNet(dim * 4)

        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=dim * 8, num_heads=heads[3],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # 注意：这里 out_channels=16，对应系数域通道
        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

        # SAM 和 merge 都在 16 通道上工作
        self.sam = SAM(out_channels, kernel_size=1, bias=bias)
        self.merge = mergeblock(out_channels, 3, True)

        # 强制初始化 SAM 和 merge 的卷积，避免旧 checkpoint 的 1 通道权重残留
        for m in [self.sam, self.merge]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, inp_img, samfeats=None, enc_in=None, dec_in=None, stage_inter=False):
        enc: List[torch.Tensor] = []
        dec: List[torch.Tensor] = []

        inp_enc_level0 = self.patch_embed(inp_img)
        inp_enc_level1 = self.CAB(inp_enc_level0)
        if samfeats is not None:
            inp_enc_level1 = self.merge(inp_enc_level1, samfeats)

        if enc_in is not None and dec_in is not None and stage_inter:
            inp_enc_level1 = self.f1_2(inp_enc_level1, enc_in[0], dec_in[-1])
        enc.append(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        if enc_in is not None and dec_in is not None and stage_inter:
            inp_enc_level2 = self.f2_3(inp_enc_level2, enc_in[1], dec_in[-2])
        enc.append(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        if enc_in is not None and dec_in is not None and stage_inter:
            inp_enc_level3 = self.f3_4(inp_enc_level3, enc_in[2], dec_in[-3])
        enc.append(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        dec.append(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        dec.append(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)          # [B,16,H,W]
        dec.append(out_dec_level1)

        samfeats, out_dec_level1 = self.sam(out_dec_level1, out_dec_level1)

        return out_dec_level1, samfeats, enc, dec
