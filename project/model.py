"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 22日 星期三 23:13:05 CST
# ***
# ************************************************************************************/
#

import math
import os
import pdb  # For debug
import sys

import torch
import torch.nn as nn
from tqdm import tqdm


def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)


def get_model(checkpoint):
    """Create model."""

    model_setenv()
    model = SegmentModel()
    model_load(model, checkpoint)
    device = model_device()
    model.to(device)

    return model


def model_device():
    """Please call after model_setenv. """

    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.environ["DEVICE"] == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])






class SegmentModel(nn.Module):
    """segment Model."""

    def __init__(self):
        """Init model."""
        super(SegmentModel, self).__init__()

    def forward(self, x):
        """Forward."""

        return x
