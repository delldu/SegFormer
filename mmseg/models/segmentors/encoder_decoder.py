import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import pdb

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
         # backbone -- {'type': 'mit_b2', 'style': 'pytorch'}
        self.backbone = builder.build_backbone(backbone)
        

        self._init_decode_head(decode_head)
        # self.decode_head
        # SegFormerHead(
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


        # self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        # assert self.with_decode_head


    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        # self.align_corners -- False
        self.num_classes = self.decode_head.num_classes
        # decode_head = {'type': 'SegFormerHead', 
        #     'in_channels': [64, 128, 320, 512], 
        #     'in_index': [0, 1, 2, 3], 
        #     'feature_strides': [4, 8, 16, 32], 
        #     'channels': 128, 'dropout_ratio': 0.1, 
        #     'num_classes': 150, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 
        #     'align_corners': False, 
        #     'decoder_params': {'embed_dim': 768}, 
        #     'loss_decode': {'type': 'CrossEntropyLoss', 
        #     'use_sigmoid': False, 'loss_weight': 1.0}}

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()


    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        # ==> pdb.set_trace()
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        x = self.backbone(img)
        # self.test_cfg -- {'mode': 'whole'}
        seg_logit = self.decode_head.forward_test(x, img_meta, self.test_cfg)
        # rescale -- True
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False,
                warning=False)

        # img.size() -- torch.Size([1, 3, 512, 704])
        # seg_pred.size() -- torch.Size([1, 150, 960, 1280])
        return F.softmax(seg_logit, dim=1)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        # pdb.set_trace() ==> here !!! 

        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # good !!!
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        pdb.set_trace()
        return seg_pred
