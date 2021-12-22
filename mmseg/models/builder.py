import warnings

from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')

import pdb

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    # isinstance(cfg, list) -- False

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        # default_args -- {'train_cfg': None, 'test_cfg': None}
        
        # {'type': 'EncoderDecoder', 'pretrained': None, 
        # 'backbone': {'type': 'mit_b2', 'style': 'pytorch'}, 
        # 'decode_head': {'type': 'SegFormerHead', 'in_channels': [64, 128, 320, 512], 
        #     'in_index': [0, 1, 2, 3], 'feature_strides': [4, 8, 16, 32], 'channels': 128, 
        #     'dropout_ratio': 0.1, 'num_classes': 150, 
        #     'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
        #     'align_corners': False, 'decoder_params': {'embed_dim': 768},
        #     'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}},
        #     'train_cfg': None, 'test_cfg': {'mode': 'whole'}}


        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    # pp SEGMENTORS
    # Registry(name=segmentor, items={'EncoderDecoder': 
    #     <class 'mmseg.models.segmentors.encoder_decoder.EncoderDecoder'>, 
    #     'CascadeEncoderDecoder': <class 'mmseg.models.segmentors.cascade_encoder_decoder.CascadeEncoderDecoder'>})


    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
