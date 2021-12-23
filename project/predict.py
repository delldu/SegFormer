"""Model predict."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 22日 星期三 23:13:05 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import get_model, model_device
from data import ADE20K

def resize_as_input(output, input):
    size = [input.size(2), input.size(3)]
    return F.interpolate(output.float(), size, mode="bilinear", align_corners=False).long()

def landmark_segment(input_tensor, output_tensor):
    palette = np.array(ADE20K.PALETTE)
    # input_tensor.size() -- [3, 512, 512]
    color_numpy = np.zeros((input_tensor.size(1), input_tensor.size(2), 3), dtype=np.uint8)
    mask_numpy = output_tensor.squeeze(0).squeeze(0).numpy()
    for label, color in enumerate(palette):
        color_numpy[mask_numpy == label, :] = color
    color_tensor = torch.from_numpy(color_numpy).permute(2, 0, 1)

    return 0.5*input_tensor.cpu() + 0.5 * color_tensor/255.0


if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="models/image_segment_b2.pth", help="checkpint file")
    parser.add_argument("--input", type=str, default="images/*.png", help="input image")
    parser.add_argument("--output", type=str, default="output", help="output folder")

    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.output):
        os.makedirs(args.output)


    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    to_tensor = T.ToTensor()
    to_image = T.ToPILImage()
    norm_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        image_tensor = to_tensor(image)

        # limition: input_tensor hxw must be divided by 32
        input_tensor = norm_tensor(image_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # input_tensor.size() torch.Size([1, 3, 512, 512])
        # pp output_tensor.size() -- torch.Size([1, 1, 128, 128])
        output_tensor = resize_as_input(output_tensor, input_tensor)

        blender_tensor = landmark_segment(image_tensor, output_tensor.cpu())
        to_image(blender_tensor).save("{}/{}".format(args.output, os.path.basename(filename)))
