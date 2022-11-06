# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed
import pdb

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        # print("MLP: ", self.proj)
        # MLP:  Linear(in_features=512, out_features=768, bias=True)
        # MLP:  Linear(in_features=320, out_features=768, bias=True)
        # MLP:  Linear(in_features=128, out_features=768, bias=True)
        # MLP:  Linear(in_features=64, out_features=768, bias=True)


    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        # (Pdb) x.size() -- torch.Size([1, 512, 16, 22])
        # (Pdb) y = x.flatten(2).transpose(1, 2)
        # (Pdb) y.size() -- torch.Size([1, 352, 512])


        # MLP: forward-input:  torch.Size([1, 512, 16, 16])
        # MLP: forward-output:  torch.Size([1, 256, 768])

        # MLP: forward-input:  torch.Size([1, 320, 32, 32])
        # MLP: forward-output:  torch.Size([1, 1024, 768])

        # MLP: forward-input:  torch.Size([1, 128, 64, 64])
        # MLP: forward-output:  torch.Size([1, 4096, 768])

        # MLP: forward-input:  torch.Size([1, 64, 128, 128])
        # MLP: forward-output:  torch.Size([1, 16384, 768])
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]


        # (Pdb) a 512x512 ade20k B2 backbone
        # self = SegFormerHead(
        #   input_transform=multiple_select, ignore_index=255, align_corners=False
        #   (loss_decode): CrossEntropyLoss()
        #   (conv_seg): Conv2d(128, 150, kernel_size=(1, 1), stride=(1, 1))
        #   (dropout): Dropout2d(p=0.1, inplace=False)
        # )
        # feature_strides = [4, 8, 16, 32]
        # kwargs = {'in_channels': [64, 128, 320, 512], 
        #     'in_index': [0, 1, 2, 3], 'channels': 128, 'dropout_ratio': 0.1, 'num_classes': 150, 
        #     'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 
        #     'align_corners': False, 'decoder_params': {'embed_dim': 768}, 
        #     'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}}

        # in_channels': [64, 128, 320, 512] ???


        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels -- (64, 128, 320, 512)

        decoder_params = kwargs['decoder_params']
        # decoder_params -- {'embed_dim': 768}
        embedding_dim = decoder_params['embed_dim']
        # embedding_dim == 256 for b1, == 768 for b2

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # (Pdb) a
        # self = SegFormerHead(
        #   input_transform=multiple_select, ignore_index=255, align_corners=False
        #   (loss_decode): CrossEntropyLoss()
        #   (conv_seg): Conv2d(128, 150, kernel_size=(1, 1), stride=(1, 1))
        #   (dropout): Dropout2d(p=0.1, inplace=False)
        #   (linear_c4): MLP(
        #     (proj): Linear(in_features=512, out_features=768, bias=True)
        #   )
        #   (linear_c3): MLP(
        #     (proj): Linear(in_features=320, out_features=768, bias=True)
        #   )
        #   (linear_c2): MLP(
        #     (proj): Linear(in_features=128, out_features=768, bias=True)
        #   )
        #   (linear_c1): MLP(
        #     (proj): Linear(in_features=64, out_features=768, bias=True)
        #   )
        #   (linear_fuse): ConvModule(
        #     (conv): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (bn): SyncBatchNorm(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (activate): ReLU(inplace=True)
        #   )
        #   (linear_pred): Conv2d(768, 150, kernel_size=(1, 1), stride=(1, 1))
        # )
        # feature_strides = [4, 8, 16, 32]
        # kwargs = {'in_channels': [64, 128, 320, 512], 
        # 'in_index': [0, 1, 2, 3], 'channels': 128, 'dropout_ratio': 0.1, 
        # 'num_classes': 150, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 
        # 'align_corners': False, 'decoder_params': {'embed_dim': 768}, 
        # 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 
        # 'loss_weight': 1.0}}

    def forward(self, inputs):
        # print("len(inputs) -- ", len(inputs))
        # for i in range(len(inputs)):
        #     print("inputs: ", i, " --- ", inputs[i].size())
        # len(inputs) --  4
        # inputs:  0  ---  torch.Size([1, 64, 128, 128])
        # inputs:  1  ---  torch.Size([1, 128, 64, 64])
        # inputs:  2  ---  torch.Size([1, 320, 32, 32])
        # inputs:  3  ---  torch.Size([1, 512, 16, 16])

        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        # print("c1, c2, c3, c4:", c1.size(), c2.size(), c3.size(), c4.size())
        # c1, c2, c3, c4: 
        # torch.Size([1, 64, 128, 128]) 
        # torch.Size([1, 128, 64, 64]) 
        # torch.Size([1, 320, 32, 32]) 
        # torch.Size([1, 512, 16, 16])

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
