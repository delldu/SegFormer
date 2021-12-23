"""Data loader."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 22日 星期三 23:13:05 CST
# ***
# ************************************************************************************/
#

import os
import pdb  # For debug

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Define Train/Test Dataset Root
# ***
# ************************************************************************************/
#
train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"


class ADE20K(object):
    """ADE20K dataset Label.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories.
    """
    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]


def grid_image(tensor_list, nrow=3):
    grid = utils.make_grid(torch.cat(tensor_list, dim=0), nrow=nrow)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image


def multiple_scale(data, multiple=32):
    """
    Scale image to a multiple.
    input data is tensor, with CxHxW format.
    """
    C, H, W = data.shape
    Hnew = ((H - 1) // multiple + 1) * multiple
    Wnew = ((W - 1) // multiple + 1) * multiple
    temp = data.new_zeros(C, Hnew, Wnew)
    temp[:, 0:H, 0:W] = data

    return temp


def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)




class segmentDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, transforms=get_transform()):
        """Init dataset."""
        super(segmentDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        self.images = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        """Load images."""

        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return img

    def __len__(self):
        """Return total numbers."""

        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """

        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of samples: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms: "
        fmt_str += "{0}{1}\n".format(tmp, self.transforms.__repr__().replace("\n", "\n" + " " * len(tmp)))
        return fmt_str


def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = segmentDataset(train_dataset_rootdir, get_transform(train=True))
    print(train_ds)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Split train_ds in train and valid set with 0.2
    # ***
    # ************************************************************************************/
    #
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl


def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = segmentDataset(test_dataset_rootdir, get_transform(train=False))
    test_dl = data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


def segmentDatasetTest():
    """Test dataset ..."""

    ds = segmentDataset(train_dataset_rootdir)
    print(ds)
    # src, tgt = ds[0]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()


if __name__ == "__main__":
    segmentDatasetTest()
