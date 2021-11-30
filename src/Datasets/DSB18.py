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

############## Helper Functions ##############


def computeDistTransform(img):
    """
    Compute the distance transform of a binary image
    :param img: binary image
    :return: distance transform
    """

    dist = cv2.distanceTransform(img.astype(np.uint8), cv2.DIST_L2, 5)
    # normalize the distance transform
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

    return dist


# vectorized_computeDistTransform = np.vectorize(
#     computeDistTransform, signature='(n),(m),()->(k)')


def computeContours(img):
    """
    Compute the contours of a binary image using the Satoshii algorithm.
    :param img: binary image
    :return: contours
    """
    contours, _ = cv2.findContours((img*255).astype(np.uint8),
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_NONE)
    return contours


# vectorized_computeDistTransform = np.vectorize(computeContours)

##############################################


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
        img = transform.resize(img, (256, 256))

        (objectProbas, overlapProba, objectContours, mask, masks) = self.get_mask(
            mask_folder, 256, 256)
        # mask = mask.astype(np.uint8)
        # mask = mask.astype(np.uint8)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        target = {}
        target["objectProbas"] = objectProbas
        target["overlapProba"] = overlapProba
        target["objectContours"] = objectContours
        target["mask"] = mask
        target["masks"] = masks

        # print(mask.shape)
        # mask = mask.permute(2, 0, 1)
        return img, target

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        """
        Get the mask of a given image from the mask folder of the dataset
        :param mask_folder: path to the mask folder
        :param IMG_HEIGHT: height of the image
        :param IMG_WIDTH: width of the image
        :return: mask
        """
        # objectProbas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=(np.uint8))
        # overlapProba = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=(np.uint8))
        objectContours = []
        # mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=(np.uint8))

        masks = np.stack([transform.resize(io.imread(os.path.join(mask_folder, mask_)).astype(np.uint8), (IMG_HEIGHT, IMG_WIDTH), 0)
                          for mask_ in os.listdir(mask_folder)], axis=0).astype(np.float32)  # read all masks

        # mask = vectorized_computeDistTransform(masks)

        #
        overlapProba = np.sum(masks, axis=0) > 1.0
        objectProbas = np.max(np.stack(tuple(computeDistTransform(
            masks[i]) for i in range(masks.shape[0])), axis=0), axis=0)
        mask = np.max(masks, axis=0)
        objectContours = [computeContours(masks[i])[0].reshape(-1, 2)
                          for i in range(masks.shape[0])]

        return (objectProbas, overlapProba, objectContours, mask, masks)


def collate_fn(batch):

    images, targets, = [], []
    for (img, target) in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


# def collate_fn(batch):
#     return tuple(zip(*batch))


class Nuclie_datamodule(pl.LightningDataModule):
    """
    Datamonodule for Nuclei dataset
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
        return DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # def val_dataloader(self):
    #     val_dataset = Nuclie_data(
    #         path=os.path.join(os.path.dirname(__file__), "DSB18/test"))
    #     return DataLoader(val_dataset, batch_size=32, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=4, shuffle=False)
