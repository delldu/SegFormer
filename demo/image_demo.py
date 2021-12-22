from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import pdb

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # pp args -- Namespace(
    #     checkpoint='models/segformer.b1.512x512.ade.160k.pth', 
    #     config='local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py', 
    #     device='cuda:0', 
    #     img='data/ADE20K_2016_07_26/images/validation/a/abbey/ADE_val_00000001.jpg', 
    #     palette='ade20k')


    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, get_palette(args.palette))


if __name__ == '__main__':
    main()
