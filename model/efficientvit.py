# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
        structure (str): 'serial' or 'parallel'.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],
                 structure='serial',
                 parallel_type='parallel',):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.structure = structure
        self.parallel_type = parallel_type

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    
    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(self.num_heads, dim=1)

        # --- [이전 하이브리드 V1 로직 시작] ---
        if self.structure == 'parallel' and self.num_heads > 1:
            # (H1->H2) || (H3) 로직 (num_heads=3일 때)

            num_groups = 2 # 2개의 병렬 그룹으로 고정
            heads_per_group = self.num_heads // num_groups
            remaining_heads = self.num_heads % num_groups

            group_defs = []
            start_idx = 0
            for g in range(num_groups):
                num_in_this_group = heads_per_group
                if g < remaining_heads: 
                    num_in_this_group += 1
                if num_in_this_group == 0:
                    continue
                group_defs.append(list(range(start_idx, start_idx + num_in_this_group)))
                start_idx += num_in_this_group

            all_feats_out = {}

            for group_heads in group_defs:
                if not group_heads: continue
                feat = feats_in[group_heads[0]]

                for i, head_idx in enumerate(group_heads):
                    if i > 0:
                        feat = feat + feats_in[head_idx]

                    feat_qkv = self.qkvs[head_idx](feat)
                    q, k, v = feat_qkv.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
                    q = self.dws[head_idx](q)
                    q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

                    q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

                    # Flash Attention Implementation
                    q = q.transpose(-2, -1) # (B, N, C)
                    k = k.transpose(-2, -1) # (B, N, C)
                    v = v.transpose(-2, -1) # (B, N, C)
                    
                    bias = (trainingab[head_idx] if self.training else self.ab[head_idx])
                    
                    # attn_mask handles the bias. SDPA supports additive mask for float inputs.
                    x_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
                    
                    feat = x_out.transpose(-2, -1).view(B, self.d, H, W)
                    all_feats_out[head_idx] = feat

            feats_out = [all_feats_out[i] for i in range(self.num_heads)]

        elif self.structure == 'parallel' and self.num_heads > 1 and self.parallel_type == 'parallel_v2':
            # (H1 || H2) -> H3 logic (for num_heads=3)
            # H1 and H2 are independent (parallel).
            # H3 takes input + H1_out + H2_out.
            
            # Assuming num_heads=3 for this specific logic as requested, 
            # or generalizing: Last head depends on all previous parallel heads.
            # Let's implement generalized version:
            # Group 1: Heads [0, ..., N-2] run in parallel.
            # Group 2: Head [N-1] runs after summing outputs of Group 1.
            
            # For num_heads=3: H0, H1 parallel. H2 takes sum of H0, H1.
            
            # 1. Process parallel heads (all except last)
            parallel_outputs = []
            accumulated_feat = 0
            
            # Parallel part
            for i in range(self.num_heads - 1):
                feat = feats_in[i]
                feat_qkv = self.qkvs[i](feat)
                q, k, v = feat_qkv.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
                q = self.dws[i](q)
                q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
                q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
                
                # Flash Attention Implementation
                q = q.transpose(-2, -1)
                k = k.transpose(-2, -1)
                v = v.transpose(-2, -1)
                bias = (trainingab[i] if self.training else self.ab[i])
                
                x_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
                feat_out = x_out.transpose(-2, -1).view(B, self.d, H, W)
                
                parallel_outputs.append(feat_out)
                accumulated_feat = accumulated_feat + feat_out
                
            # 2. Process last head (dependent on parallel heads)
            last_idx = self.num_heads - 1
            feat = feats_in[last_idx] + accumulated_feat # Add accumulated outputs from previous layer
            
            feat_qkv = self.qkvs[last_idx](feat)
            q, k, v = feat_qkv.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[last_idx](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            
            # Flash Attention Implementation
            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            bias = (trainingab[last_idx] if self.training else self.ab[last_idx])
            
            x_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
            feat_out = x_out.transpose(-2, -1).view(B, self.d, H, W)
            
            feats_out = parallel_outputs + [feat_out]

        else: 
            # 'serial' (원본 로직)
            feats_out = []
            feat = feats_in[0]
            for i, qkv in enumerate(self.qkvs):
                if i > 0: # add the previous output to the input
                    feat = feat + feats_in[i]
                feat = qkv(feat)
                q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
                q = self.dws[i](q)
                q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
                q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
                
                # Flash Attention Implementation
                q = q.transpose(-2, -1)
                k = k.transpose(-2, -1)
                v = v.transpose(-2, -1)
                bias = (trainingab[i] if self.training else self.ab[i])
                
                x_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
                feat = x_out.transpose(-2, -1).view(B, self.d, H, W)
                
                feats_out.append(feat)
        # --- [로직 수정 끝] ---

        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],
                 attn_structure='serial',
                 parallel_type='parallel',):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,
                                structure=attn_structure,
                                parallel_type=parallel_type,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            
            # --- [이 self.attn(x) 호출이 CascadedGroupAttention.forward()를 실행합니다] ---
            x = self.attn(x) 
            
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):    
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],
                 attn_structure='serial',
                 parallel_type='parallel',):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
            resolution=resolution, window_resolution=window_resolution, kernels=kernels,
            attn_structure=attn_structure, parallel_type=parallel_type))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,
                 parallel_stages=None,
                 parallel_type='parallel',):
        super().__init__()

        resolution = img_size
        #         Conv2d_BN(in_chans, embed_dim[0] // 8, ks=3, stride=1, pad=1, resolution=resolution),
        #         torch.nn.ReLU(),
        #         Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, ks=3, stride=2, pad=1, resolution=resolution),
        #         torch.nn.ReLU(),
        #         Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, ks=3, stride=2, pad=1, resolution=resolution // 2),
        #         torch.nn.ReLU(),
        #         Conv2d_BN(embed_dim[0] // 2, embed_dim[0], ks=3, stride=1, pad=1, resolution=resolution // 4)
        #     )
        #    resolution = img_size // patch_size

        if img_size == 32:
            # 32x32 -> 16x16 (2x downsampling) - [ks, pad 인자 이름 수정]
            self.patch_embed = torch.nn.Sequential(
                Conv2d_BN(in_chans, embed_dim[0] // 2, ks=3, stride=1, pad=1, resolution=resolution),
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], ks=3, stride=2, pad=1, resolution=resolution // 2),
            )
            resolution = img_size // 2  # 16
        
            
        elif img_size == 224:  # ImageNet (원본)
            # 224 -> 112 -> 56 -> 28 -> 14
            self.patch_embed = torch.nn.Sequential(
                Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8)
            )
            resolution = img_size // patch_size  # 224 // 16 = 14
            
        else:
            raise ValueError(f"Unsupported img_size={img_size} and patch_size={patch_size} combination")

        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                current_attn_structure = 'serial' # 기본값은 직렬
                if parallel_stages is not None and i in parallel_stages:
                    current_attn_structure = 'parallel'
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels, attn_structure=current_attn_structure, parallel_type=parallel_type))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x
