"""Create model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022-2024, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 27日 星期二 02:00:52 CST
# ***
# ************************************************************************************/
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from typing import Tuple
from functools import partial
import todos
import pdb  # For debug

FEATURE_RESULT_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def torch_nn_arange(x):
    if x.dim() == 2:
        B, C = x.size()
        a = torch.arange(x.nelement())/x.nelement()
        a = a.to(x.device)
        return a.view(B, C)

    if x.dim() == 3:
        B, C, HW = x.size()
        a = torch.arange(x.nelement())/x.nelement()
        a = a.to(x.device)
        return a.view(B, C, HW)

    B, C, H, W = x.size()
    a = torch.arange(x.nelement())/x.nelement()
    a = a.to(x.device)
    return a.view(B, C, H, W)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H: int, W: int):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class BlockMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, H: int, W: int):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads # [1, 2, 5, 8]
        self.scale = 64 ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio # maybe 8, 4, 2, 1
        if sr_ratio > 1: # [8, 4, 2, 1]
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:  # Support torch.jit.script
            self.sr = nn.Identity()
            self.norm = nn.Identity()
        # Attention(
        #   (q): Linear(in_features=320, out_features=320, bias=True)
        #   (kv): Linear(in_features=320, out_features=640, bias=True)
        #   (proj): Linear(in_features=320, out_features=320, bias=True)
        #   (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
        #   (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        # )
        assert self.scale == 0.125

    def forward(self, x, H: int, W: int):
        B, HW, C = x.shape
        assert HW == H*W

        q = self.q(x).reshape(B, HW, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q -- [B, num_heads, HW, C//num_heads]
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1) # x-[B, HW, C]

            x = self.norm(x)

        kv = self.kv(x).reshape(B, -1, self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        # kv -- [B, num_heads, HW, C//num_heads]
        N2 = kv.size(2)
        k = kv[:, :, 0:N2:2, :]
        v = kv[:, :, 1:N2:2, :]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # tensor [attn] size: [1, 5, 4800, 1200], min: 0.0, max: 0.718379, mean: 0.000833
        # tensor [v] size: [1, 5, 1200, 64], min: -2.922277, max: 2.865681, mean: -0.005795
        # tensor [attn@v] size: [1, 5, 4800, 64], min: -1.528984, max: 1.475878, mean: -0.001349
        # --------------------------------------------------------------------------------
        x = (attn @ v).transpose(1, 2).reshape(B, HW, C)

        x = self.proj(x) # [B, HW, C]
        assert HW == H*W

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BlockMlp(in_features=dim, hidden_features=dim * 4)

    def forward(self, x, H: int, W: int):
        # x -- [B, HW, C]
        # tensor [x] size: [1, 15360, 128], min: -5.810387, max: 4.976694, mean: 0.02447
        x = x + self.attn(self.norm1(x), H, W)

        # todos.debug.output_var("mlp", x)

        x = x + self.mlp(self.norm2(x), H, W)
        # tensor [x] size: [1, 3840, 320], min: -14.541649, max: 63.599552, mean: -0.086237

        return x # [B, HW, C] 


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        todos.debug.output_var("==> x1", x)
        proj_out = x
        # tensor [x] size: [1, 64, 240, 320], min: -8.812546, max: 10.21946, mean: -0.002233
        # tensor [x] size: [1, 128, 120, 160], min: -10.213951, max: 6.449842, mean: -0.02391

        x = x.flatten(2).transpose(1, 2) # (B, C, HW) -> (B, HW, C)
        x = self.norm(x)
        # tensor [x1] size: [1, 3, 960, 1280], min: -2.117904, max: 2.64, mean: 0.034018
        # tensor [x2] size: [1, 76800, 64], min: -2.877892, max: 2.634512, mean: -0.018559
        # --------------------------------------------------------------------------------
        # tensor [x1] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
        # tensor [x2] size: [1, 19200, 128], min: -5.916475, max: 5.062787, mean: 0.023141
        # --------------------------------------------------------------------------------
        # tensor [x1] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
        # tensor [x2] size: [1, 4800, 320], min: -6.143896, max: 6.973696, mean: -0.089654
        # --------------------------------------------------------------------------------
        # tensor [x1] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
        # tensor [x2] size: [1, 1200, 512], min: -14.498089, max: 143.741379, mean: -0.082211
        # --------------------------------------------------------------------------------        

        todos.debug.output_var("==> x2", x)

        return (x, proj_out)


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # patch_embed
        embed_dims=[64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        num_heads=[1, 2, 5, 8]
        depths= [3, 8, 27, 3]
        sr_ratios=[8, 4, 2, 1]
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])
        # pdb.set_trace()

    def forward(self, x) -> FEATURE_RESULT_TYPE:
        B = x.shape[0]

        # stage 1
        x, x_proj_out = self.patch_embed1(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        todos.debug.output_var(">x", x)

        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x1 = x

        # stage 2
        x, x_proj_out = self.patch_embed2(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x

        # stage 3
        x, x_proj_out = self.patch_embed3(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x

        # stage 4
        x, x_proj_out = self.patch_embed4(x)
        _, _, H, W = x_proj_out.shape
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x4 = x

        # tensor [>x] size: [1, 3, 960, 1024], min: -2.117904, max: 2.64, mean: 0.034037
        # tensor [x1] size: [1, 64, 240, 256], min: -4.195707, max: 4.108228, mean: 0.012766
        # tensor [x2] size: [1, 128, 120, 128], min: -6.148735, max: 5.365499, mean: 0.023484
        # tensor [x3] size: [1, 320, 60, 64], min: -5.416265, max: 4.744856, mean: -0.002299
        # tensor [x4] size: [1, 512, 30, 32], min: -6.56222, max: 7.764719, mean: 0.025833

        return (x1, x2, x3, x4)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, 768)

    def forward(self, x):
        # tensor [MLP: x] size: [1, 512, 30, 40], min: -6.624208, max: 8.50036, mean: 0.025605
        # tensor [MLP: x] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
        # tensor [MLP: x] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
        # tensor [MLP: x] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers."""
    def __init__(self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(self.bn(x))
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 150  # for ADE20K dataset

        self.conv_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.linear_c1 = MLP(input_dim=64)
        self.linear_c2 = MLP(input_dim=128)
        self.linear_c3 = MLP(input_dim=320)
        self.linear_c4 = MLP(input_dim=512)

        self.linear_fuse = ConvModule(in_channels=768 * 4, out_channels=768)
        self.linear_pred = nn.Conv2d(768, self.num_classes, kernel_size=1)

    def forward(self, inputs: FEATURE_RESULT_TYPE):
        # inputs is tuple: len = 4
        #     tensor [item] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
        #     tensor [item] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
        #     tensor [item] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
        #     tensor [item] size: [1, 512, 30, 40], min: -6.624208, max: 8.50036, mean: 0.025605

        x = inputs  # len=4, 1/4,1/8,1/16,1/32
        # c1, c2, c3, c4 = x # onnx not happy
        c1 = x[0]
        c2 = x[1]
        c3 = x[2]
        c4 = x[3]

        ############## MLP decoder on C1-C4 ###########
        B1, C1, H1, W1 = c1.size()
        B2, C2, H2, W2 = c2.size()
        B3, C3, H3, W3 = c3.size()
        B4, C4, H4, W4 = c4.size()

        _c4 = self.linear_c4(c4).reshape(B4, -1, H4, W4)
        _c4 = F.interpolate(_c4, size=(H1, W1), mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3).reshape(B4, -1, H3, W3)
        _c3 = F.interpolate(_c3, size=(H1, W1), mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2).reshape(B4, -1, H2, W2)
        _c2 = F.interpolate(_c2, size=(H1, W1), mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1).reshape(B4, -1, H1, W1)

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.linear_pred(x)

        return x


class SegmentModel(nn.Module):
    """Encoder Decoder segmentors."""

    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 2048
        self.MAX_TIMES = 4
        # GPU 1024x1024 -- 4G, 120ms
        # GPU 1024x2048 -- 7.6G, 640ms

        self.backbone = VisionTransformer()
        self.decode_head = SegFormerHead() #self.backbone.embedding_dim)
        self.num_classes = self.decode_head.num_classes # 150

        self.load_weights()
        self.eval()
        # pdb.set_trace()


    def load_weights(self, model_path="models/image_segment.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)['state_dict']

        self.load_state_dict(sd)
        # from ggml_engine import create_network
        # create_network(self)
        # torch.save(self.state_dict(), "/tmp/a.pth")

    def forward(self, x):
        B, C, H, W = x.shape
        # x.size() -- ([1, 3, 960, 1280])
        r_pad = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        b_pad = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        x = F.pad(x, (0, r_pad, 0, b_pad), mode="replicate")

        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        todos.debug.output_var("images", x)

        f = self.backbone(x)
        todos.debug.output_var("f", f)

        seg_logit = self.decode_head(f)
        todos.debug.output_var("seg_decoder", seg_logit)

        seg_logit = F.interpolate(seg_logit, size=(H,W), mode="bilinear", align_corners=False)
        seg_logit = F.softmax(seg_logit, dim=1) # [1, 150, 960, 1280]
        todos.debug.output_var("seg_logit", seg_logit)


        mask = seg_logit.argmax(dim=1).unsqueeze(0) # [1, 960, 1280] -> [1, 1, 960, 1280]
        todos.debug.output_var("seg_mask", mask)

        # mask = mask[:, :, 0:H, 0:W]
        mask = mask.clamp(0, self.num_classes).to(torch.float32)
        # ADE20K class number is 150, to float32 is for onnx export
        # tensor [mask] size: [1, 1, 960, 1280], min: 0.0, max: 25.0, mean: 3.283945        

        return mask
