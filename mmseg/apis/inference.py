import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import pdb

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    # (Pdb) pp config.model
    # {'backbone': {'style': 'pytorch', 'type': 'mit_b1'},
    #  'decode_head': {'align_corners': False,
    #                  'channels': 128,
    #                  'decoder_params': {'embed_dim': 256},
    #                  'dropout_ratio': 0.1,
    #                  'feature_strides': [4, 8, 16, 32],
    #                  'in_channels': [64, 128, 320, 512],
    #                  'in_index': [0, 1, 2, 3],
    #                  'loss_decode': {'loss_weight': 1.0,
    #                                  'type': 'CrossEntropyLoss',
    #                                  'use_sigmoid': False},
    #                  'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
    #                  'num_classes': 150,
    #                  'type': 'SegFormerHead'},
    #  'pretrained': None,
    #  'test_cfg': {'mode': 'whole'},
    #  'train_cfg': None,
    #  'type': 'EncoderDecoder'}
    # checkpoint -- 'models/segformer.b1.512x512.ade.160k.pth'

    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']


    # (Pdb) len(model.CLASSES) -- 150
    # model.CLASSES -- ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 
    #     'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 
    #     'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 
    #     'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 
    #     'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 
    #     'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 
    #     'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 
    #     'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 
    #     'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 
    #     'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 
    #     'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 
    #     'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 
    #     'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 
    #     'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 
    #     'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 
    #     'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 
    #     'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 
    #     'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 
    #     'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 
    #     'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 
    #     'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 
    #     'shower', 'radiator', 'glass', 'clock', 'flag')

    # (Pdb) len(model.PALETTE) -- 150
    # model.PALETTE -- [[120, 120, 120], [180, 120, 120],  ..., [92, 0, 255]]

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # type(img) -- <class 'numpy.ndarray'> -- (Pdb) img.shape -- (960, 1280, 3)
        # img.min(), img.max() -- (0, 255)

        return results


def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # test_pipeline
    # Compose(
    #     <mmseg.apis.inference.LoadImage object at 0x7f5e8013ff98>
    #     MultiScaleFlipAug(transforms=Compose(
    #     AlignedResize(img_scale=None, multiscale_mode=range, ratio_range=None, keep_ratio=True)
    #     RandomFlip(prob=None)
    #     Normalize(mean=[123.675 116.28  103.53 ], std=[58.395 57.12  57.375], to_rgb=True)
    #     ImageToTensor(keys=['img'])
    #     Collect(keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))
    # ), img_scale=[(2048, 512)], flip=False)flip_direction=['horizontal']

    # prepare data
    # img -- 'data/ADE20K_2016_07_26/images/validation/a/abbey/ADE_val_00000001.jpg'
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # type(data) -- <class 'dict'> -- data.keys() -- dict_keys(['img_metas', 'img'])

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]


    # data
    # {'img_metas': [[{'filename': 'data/ADE20K_2016_07_26/images/validation/a/abbey/ADE_val_00000001.jpg', 
    # 'ori_filename': 'data/ADE20K_2016_07_26/images/validation/a/abbey/ADE_val_00000001.jpg', 
    # 'ori_shape': (960, 1280, 3), 'img_shape': (512, 704, 3), 'pad_shape': (512, 704, 3), 
    # 'scale_factor': array([0.55      , 0.53333336, 0.55      , 0.53333336], dtype=float32), 
    # 'flip': False, 'flip_direction': 'horizontal', 
    # 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
    # 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}]], 

    # 'img': [tensor([[[[-0.6794, -0.6965, -0.6623,  ..., -0.8164, -0.7479, -0.8164],
    #           [-0.6452, -0.7137, -0.6965,  ..., -0.8678, -0.7993, -0.7993],
    #           [-0.6281, -0.7137, -0.7137,  ..., -0.7479, -0.7479, -0.8164],
    #           ...,
    #           ...,
    #           [-0.7761, -0.7238, -0.8110,  ...,  0.0256,  0.0256,  0.0082],
    #           [-0.6890, -0.6018, -0.7064,  ..., -0.0267,  0.0605,  0.0256],
    #           [-0.6367, -0.7064, -0.6018,  ...,  0.0082,  0.0953,  0.0431]]]],
    #        device='cuda:0')]}

    # file data/ADE20K_2016_07_26/images/validation/a/abbey/ADE_val_00000001.jpg  ... 1280x960, frames 3
    # 512 == 960 * 0.53333336, 704 == 1280 * 0.55
    # data['img'][0].size() -- torch.Size([1, 3, 512, 704])


    # (Pdb) model

    # model.backbone.
    # EncoderDecoder(
    #   (backbone): mit_b1(
    #     (patch_embed1): OverlapPatchEmbed(
    #       (proj): Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
    #       (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    #     )
    #     (patch_embed2): OverlapPatchEmbed(
    #       (proj): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #       (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    #     )
    #     (patch_embed3): OverlapPatchEmbed(
    #       (proj): Conv2d(128, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #       (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
    #     )
    #     (patch_embed4): OverlapPatchEmbed(
    #       (proj): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #       (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    #     )
    #     (block1): ModuleList(
    #       (0): Block(
    #         (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=64, out_features=64, bias=True)
    #           (kv): Linear(in_features=64, out_features=128, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=64, out_features=64, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #           (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
    #           (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    #         )
    #         (drop_path): Identity()
    #         (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=64, out_features=256, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=256, out_features=64, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #       (1): Block(
    #         (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=64, out_features=64, bias=True)
    #           (kv): Linear(in_features=64, out_features=128, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=64, out_features=64, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #           (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
    #           (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=64, out_features=256, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=256, out_features=64, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #     )
    #     (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    #     (block2): ModuleList(
    #       (0): Block(
    #         (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=128, out_features=128, bias=True)
    #           (kv): Linear(in_features=128, out_features=256, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=128, out_features=128, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #           (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
    #           (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=128, out_features=512, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=512, out_features=128, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #       (1): Block(
    #         (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=128, out_features=128, bias=True)
    #           (kv): Linear(in_features=128, out_features=256, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=128, out_features=128, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #           (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
    #           (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=128, out_features=512, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=512, out_features=128, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #     )
    #     (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    #     (block3): ModuleList(
    #       (0): Block(
    #         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=320, out_features=320, bias=True)
    #           (kv): Linear(in_features=320, out_features=640, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=320, out_features=320, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
    #           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=320, out_features=1280, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=1280, out_features=320, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #       (1): Block(
    #         (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=320, out_features=320, bias=True)
    #           (kv): Linear(in_features=320, out_features=640, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=320, out_features=320, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #           (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
    #           (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=320, out_features=1280, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=1280, out_features=320, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #     )
    #     (norm3): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
    #     (block4): ModuleList(
    #       (0): Block(
    #         (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=512, out_features=512, bias=True)
    #           (kv): Linear(in_features=512, out_features=1024, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=512, out_features=512, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=512, out_features=2048, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=2048, out_features=512, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #       (1): Block(
    #         (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    #         (attn): Attention(
    #           (q): Linear(in_features=512, out_features=512, bias=True)
    #           (kv): Linear(in_features=512, out_features=1024, bias=True)
    #           (attn_drop): Dropout(p=0.0, inplace=False)
    #           (proj): Linear(in_features=512, out_features=512, bias=True)
    #           (proj_drop): Dropout(p=0.0, inplace=False)
    #         )
    #         (drop_path): DropPath()
    #         (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    #         (mlp): Mlp(
    #           (fc1): Linear(in_features=512, out_features=2048, bias=True)
    #           (dwconv): DWConv(
    #             (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
    #           )
    #           (act): GELU()
    #           (fc2): Linear(in_features=2048, out_features=512, bias=True)
    #           (drop): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #     )
    #     (norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    #   )

    # model.decode_head
    #   (decode_head): SegFormerHead(
    #     input_transform=multiple_select, ignore_index=255, align_corners=False
    #     (loss_decode): CrossEntropyLoss()
    #     (conv_seg): Conv2d(128, 150, kernel_size=(1, 1), stride=(1, 1))
    #     (dropout): Dropout2d(p=0.1, inplace=False)
    #     (linear_c4): MLP(
    #       (proj): Linear(in_features=512, out_features=256, bias=True)
    #     )
    #     (linear_c3): MLP(
    #       (proj): Linear(in_features=320, out_features=256, bias=True)
    #     )
    #     (linear_c2): MLP(
    #       (proj): Linear(in_features=128, out_features=256, bias=True)
    #     )
    #     (linear_c1): MLP(
    #       (proj): Linear(in_features=64, out_features=256, bias=True)
    #     )
    #     (linear_fuse): ConvModule(
    #       (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (activate): ReLU(inplace=True)
    #     )
    #     (linear_pred): Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
    #   )
    # )

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    # (Pdb) len(result) -- 1
    # (Pdb) type(result[0]) -- <class 'numpy.ndarray'>
    # (Pdb) result[0].shape -- (960, 1280)
    # (Pdb) result[0]
    # array([[2, 2, 2, ..., 2, 2, 2],
    #        [2, 2, 2, ..., 2, 2, 2],
    #        [2, 2, 2, ..., 2, 2, 2],
    #        ...,
    #        [9, 9, 9, ..., 6, 6, 6],
    #        [9, 9, 9, ..., 6, 6, 6],
    #        [9, 9, 9, ..., 6, 6, 6]])
    # result[0].min(), result[0].max() -- (0, 52)

    # new test case 
    # data['img'][0].size() -- torch.Size([1, 3, 512, 512])
    #  data['img']
    # [tensor([[[[ 1.8037,  1.8379,  1.9064,  ...,  1.6495,  1.6324,  1.6495],
    #           [ 1.8037,  1.8550,  1.9064,  ...,  1.6495,  1.6324,  1.5982],
    #           [ 1.8208,  1.8722,  1.9235,  ...,  1.6324,  1.5639,  1.5297],
    #           ...,
    #           ...,
    #           [-0.6367, -0.6367, -0.7064,  ..., -1.0724, -0.8633, -0.8807],
    #           [-0.8633, -0.8981, -0.7761,  ..., -1.0724, -0.9853, -0.8807],
    #           [-0.9156, -1.0027, -0.8633,  ..., -0.8110, -0.9853, -1.0376]]]],
    #        device='cuda:0')]

    # result[0].shape -- (512, 512)

    return result


def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10)):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
    """
    # hasattr(model, 'module') -- False
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, palette=palette, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()
