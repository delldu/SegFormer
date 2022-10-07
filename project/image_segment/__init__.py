"""Image Weather Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import numpy as np

import redos
import todos

from . import segment
from . import ade20k

import pdb


def get_model():
    """Create model."""

    model_path = "models/image_segment.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    model = segment.SegmentModel()
    todos.model.load(model, checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_segment.torch"):
        model.save("output/image_segment.torch")

    return model, device


def blender_segment(input_tensor, output_tensor):
    palette = np.array(ade20k.ADE20K.PALETTE)
    B, C, H, W = input_tensor.size()

    # input_tensor.size() -- [1, 3, 512, 512]
    color_numpy = np.zeros((H, W, 3), dtype=np.uint8)
    mask_numpy = output_tensor.squeeze(0).squeeze(0).numpy()
    for label, color in enumerate(palette):
        color_numpy[mask_numpy == label, :] = color
    color_tensor = torch.from_numpy(color_numpy).permute(2, 0, 1).unsqueeze(0)

    return 0.5 * input_tensor.cpu() + 0.5 * color_tensor / 255.0


def model_forward(model, device, input_tensor, multi_times=1):
    # zeropad for model
    B, C, H, W = input_tensor.shape
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    output_tensor = todos.model.forward(model, device, input_tensor)
    final_tensor = blender_segment(input_tensor.cpu(), output_tensor.cpu())

    return final_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.segment(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, HOST="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  Segment {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "image_segment", do_service, HOST, port)


def image_predict(input_files, output_dir):
    print(f"Segment predict {input_files} ... ")

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # Load model
    model, device = get_model()

    # Load files
    image_filenames = todos.data.load_files(input_files)

    # Start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # Original input
        input_tensor = todos.data.load_tensor(filename)
        # Pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    model, device = get_model()

    print(f"  Segment {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def segment_video_frame(no, data):
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # Cnvert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=segment_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.segment(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_segment", video_service, HOST, port)


def video_predict(input_file, output_file):
    return video_service(input_file, output_file, None)
