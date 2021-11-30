import os
import time
import copy
from collections import defaultdict
import torch
import shutil
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
# from torchvision.transforms import (RandomHorizontalFlip, Normalize, Resize, Compose)
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2

import cv2
# from torchvision.transforms import
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
from torch import nn
import zipfile
import pytorch_lightning as pl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_transforms(mean, std):
    list_transforms = []

    list_transforms.extend(
        [
            HorizontalFlip(p=0.5),  # only horizontal flip as of now
        ])
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    list_trfms = Compose(list_transforms)
    return list_trfms


class Nuclie_data(Dataset):
    def __init__(self, path):
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_transforms(0.5, 0.5)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (128, 128))

        mask = self.get_mask(mask_folder, 128, 128).astype('float32')

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        # print(mask.shape)
        mask = mask.permute(2, 0, 1)
        return (img, mask)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        return mask


def mask_convert(mask):
    mask = mask.clone().cpu().detach().numpy()
    mask = mask.transpose((1, 2, 0))
    std = np.array((0.5))
    mean = np.array((0.5))
    mask = std * mask + mean
    mask = mask.clip(0, 1)
    mask = np.squeeze(mask)
    return mask

# converting tensor to image


def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1, 2, 0))
    std = np.array((0.5, 0.5, 0.5))
    mean = np.array((0.5, 0.5, 0.5))
    image = std * image + mean
    image = image.clip(0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def plot_img(no_, dataset):
    # iter_ = iter(loader)
    # images,masks = next(iter_)
    # images = images.to(device)
    # masks = masks.to(device)

    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    fig, axes = plt.subplots(nrows=2, ncols=no_, figsize=(20, 10))
    for i, id in enumerate(idx[:no_]):
        image, mask = dataset[id]
        image = image_convert(image)
        mask = mask_convert(mask)

        axes[0][i].set_title('image')
        axes[0][i].imshow(image)
        axes[1][i].set_title('mask')
        axes[1][i].imshow(mask, cmap='gray')

    fig.tight_layout()
    plt.show()


class Nuclie_datamodule(pl.LightningDataModule):
    """
    """

    def __init__(self):
        super().__init__()

    # def setup(self, stage=None):
    #     self.train_dataset = Nuclie_data(
    #         path=os.path.join(os.path.dirname(__file__), "DSB18/train"))
    #     self.val_dataset = Nuclie_data(
    #         path=os.path.join(os.path.dirname(__file__), "DSB18/test"))

    def train_dataloader(self):
        train_dataset = Nuclie_data(
            path=os.path.join(os.path.dirname(__file__), "DSB18/train"))
        return DataLoader(train_dataset, batch_size=32, shuffle=True)

    # def val_dataloader(self):
    #     val_dataset = Nuclie_data(
    #         path=os.path.join(os.path.dirname(__file__), "DSB18/test"))
    #     return DataLoader(val_dataset, batch_size=32, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=4, shuffle=False)
