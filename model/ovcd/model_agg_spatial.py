import torch
import math

from torch import nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F


TORCH_VERSION = torch.__version__

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU6(inplace=True)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.dwconv = DWConv(out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.act(x1) * x2
        x = self.fc3(x)
        x = self.dwconv(x, H, W)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class Attention(nn.Module):
    """Refer form SegFormer"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.q = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.conv_diff = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm_q = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.k1 = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.v1 = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim*2, dim, bias=qkv_bias)
        
        if TORCH_VERSION < '2.2.0':
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = attn_drop
        
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        self.conv_dr = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True)
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1: # shrink ratio
            self.sr_k1 = nn.Conv2d(2*dim, 2*dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k1 = nn.LayerNorm(2*dim)

            self.sr_k2 = nn.Conv2d(2*dim, 2*dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k2 = nn.LayerNorm(2*dim)
            
            self.sr_v1 = nn.Conv2d(2*dim, 2*dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_v1 = nn.LayerNorm(2*dim)

            self.sr_v2 = nn.Conv2d(2*dim, 2*dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_v2 = nn.LayerNorm(2*dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, image_guidance1, image_guidance2, H, W): # x: diff, image_guidance: image feature
        B, N, C = x.shape
        x_f1 = torch.cat([x, image_guidance1], dim=-1)
        x_f2 = torch.cat([x, image_guidance2], dim=-1)

        # query  diff
        x_q = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_q = self.conv_diff(x_q).reshape(B, C, -1).permute(0, 2, 1)
        x_q = self.norm_q(x_q)
        q = self.q(x_q).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # key1
            x_k1 = x_f1.permute(0, 2, 1).reshape(B, 2*C, H, W)
            x_k1 = self.sr_k1(x_k1).reshape(B, 2*C, -1).permute(0, 2, 1)
            x_k1 = self.norm_k1(x_k1)
            k1 = self.k1(x_k1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # key2
            x_k2 = x_f2.permute(0, 2, 1).reshape(B, 2*C, H, W)
            x_k2 = self.sr_k2(x_k2).reshape(B, 2*C, -1).permute(0, 2, 1)
            x_k2 = self.norm_k2(x_k2)
            k2 = self.k2(x_k2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # value1
            x_v1 = x_f1.permute(0, 2, 1).reshape(B, 2*C, H, W)
            x_v1 = self.sr_v1(x_v1).reshape(B, 2*C, -1).permute(0, 2, 1)  # (B, 2C, H/2, W/2)
            x_v1 = self.norm_v1(x_v1)
            v1 = self.v1(x_v1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # value2
            x_v2 = x_f2.permute(0, 2, 1).reshape(B, 2*C, H, W)
            x_v2 = self.sr_v2(x_v2).reshape(B, 2*C, -1).permute(0, 2, 1)  # (B, 2C, H/2, W/2)
            x_v2 = self.norm_v2(x_v2)
            v2 = self.v2(x_v2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x_k1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_v1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if TORCH_VERSION < '2.2.0':
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        else:
            out1 = F.scaled_dot_product_attention(q, k1, v1, dropout_p=self.attn_drop, scale=self.scale)
            out2 = F.scaled_dot_product_attention(q, k2, v2, dropout_p=self.attn_drop, scale=self.scale)

        out1 = out1.transpose(1, 2).reshape(B, N, C).contiguous()
        out2 = out2.transpose(1, 2).reshape(B, N, C).contiguous()

        out1 = out1.permute(0, 2, 1).reshape(B, C, H, W)
        out2 = out2.permute(0, 2, 1).reshape(B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        out = torch.cat([x, out1, out2], dim=1)
        out = self.conv_dr(out)

        out = rearrange(out, 'B C H W -> B (H W) C')

        return out


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qk = self.qk(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        v = self.v(g).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

# MoE with gating network
class MoEFFN_Gating(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFN_Gating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        ) for _ in range(num_experts)])

    def forward(self, x):
        # Gating mechanism to determine the mixture weights
        weights = self.gating_network(x)
        weights = torch.nn.functional.softmax(weights, dim=-1)

        # Log the weights
        # self.log_weights(weights)

        # Get outputs from all experts
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)

        # combine the experts' outputs
        # print(weight)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs


class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0., sr_ratio=1,
                 attn_drop=0., drop=0., qkv_bias=True, norm_layer=nn.LayerNorm):
        super(MiTBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attention = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=3*dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward1(self, diff, guidance):
        B, C, H, W = diff.shape
        diff = rearrange(diff, 'B C H W -> B (H W) C')
        guidance = rearrange(guidance, 'B C H W -> B (H W) C')
        diff = diff + self.drop_path(self.attention(self.norm1(diff), guidance, H, W))
        diff = diff + self.drop_path(self.mlp(self.norm2(diff), H, W))
        diff = rearrange(diff, 'B (H W) C -> B C H W', H=H, W=W)
        return diff
    
    def forward(self, diff, guidance1, guidance2):
        B, C, H, W = diff.shape
        diff = rearrange(diff, 'B C H W -> B (H W) C')
        guidance1 = rearrange(guidance1, 'B C H W -> B (H W) C')
        guidance2 = rearrange(guidance2, 'B C H W -> B (H W) C')
        diff = diff + self.drop_path(self.attention(self.norm1(diff), guidance1, guidance2, H, W))
        diff = diff + self.drop_path(self.mlp(self.norm2(diff), H, W))   # 在这加MoE
        diff = rearrange(diff, 'B (H W) C -> B C H W', H=H, W=W)
        return diff

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MiTBlock_Swin(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0., sr_ratio=1,
                 attn_drop=0., drop=0., qkv_bias=True, norm_layer=nn.LayerNorm):
        super(MiTBlock_Swin, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.window_size = 9
        self.num_experts = 4
        self.attention = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_g1 = norm_layer(dim)
        self.norm_g2 = norm_layer(dim)

        self.conv_g1 = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv_g2 = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1, padding=0)

        self.conv_dr = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True)
        )

        self.mlp = MoEFFN_Gating(dim=dim, hidden_dim=3*dim, num_experts=self.num_experts)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, diff, guidance1, guidance2):
        B, C, H, W = diff.shape
        diff = rearrange(diff, 'B C H W -> B (H W) C')
        guidance1 = rearrange(guidance1, 'B C H W -> B (H W) C')
        guidance2 = rearrange(guidance2, 'B C H W -> B (H W) C')

        guidance1 = torch.cat([diff, guidance1], dim=-1).permute(0, 2, 1).reshape(B, 2*C, H, W)
        guidance2 = torch.cat([diff, guidance2], dim=-1).permute(0, 2, 1).reshape(B, 2*C, H, W)

        guidance1 = self.conv_g1(guidance1).reshape(B, C, -1).permute(0, 2, 1)
        guidance2 = self.conv_g2(guidance2).reshape(B, C, -1).permute(0, 2, 1)

        shortcut = diff
        diff_w = self.norm1(diff)
        diff_w = diff_w.view(B, H, W, C)

        g1 = self.norm_g1(guidance1)
        g2 = self.norm_g2(guidance2)
        g1 = g1.view(B, H, W, C)
        g2 = g2.view(B, H, W, C)

        # partition windows
        d_w = window_partition(diff_w, self.window_size)  # nW*B, window_size, window_size, C
        d_w = d_w.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        g1 = window_partition(g1, self.window_size)
        g2 = window_partition(g2, self.window_size)
        g1 = g1.view(-1, self.window_size * self.window_size, C)
        g2 = g2.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        d_g1_w = self.attention(d_w, g1)
        d_g2_w = self.attention(d_w, g2)

        # merge windows
        d_g1_w = d_g1_w.view(-1, self.window_size, self.window_size, C)
        d_g1_w = window_reverse(d_g1_w, self.window_size, H, W)
        d_g2_w = d_g2_w.view(-1, self.window_size, self.window_size, C)
        d_g2_w = window_reverse(d_g2_w, self.window_size, H, W)

        d_g1_w = rearrange(d_g1_w, 'B H W C -> B C H W')
        d_g2_w = rearrange(d_g2_w, 'B H W C -> B C H W')

        d_w = torch.cat([d_g1_w, d_g2_w], dim=1)
        d_w = self.conv_dr(d_w)

        d_w = rearrange(d_w, 'B C H W -> B (H W) C')

        out = shortcut + self.drop_path(d_w)      # residual add
        out = out + self.drop_path(self.mlp(self.norm2(out)))

        out = rearrange(out, 'B (H W) C -> B C H W', H=H, W=W)

        return out


class SptialFusionBlock(nn.Module):
    def __init__(self, inter_channels, num_heads, drop_path=0., sr_ratio=1,):
        super(SptialFusionBlock, self).__init__()
        self.sa = MiTBlock_Swin(inter_channels, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)
        
    def forward(self, diff, img_guidance1, img_guidance2):
        spatial_feat = self.sa(diff, img_guidance1, img_guidance2)
        return spatial_feat