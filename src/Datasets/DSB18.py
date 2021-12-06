import os
import torch
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
from PIL import Image
from torch import nn
import pytorch_lightning as pl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_transforms(mean, std):
    """
    Define the transformations to be used for the dataset
    :param mean: mean of the dataset
    :param std: standard deviation of the dataset
    :return: transformations
    """
    list_transforms = [
        Normalize(mean=mean, std=std),
        ToTensorV2()
    ]

    return Compose(list_transforms)

############## Helper Functions ##############


def fillHoles(mask): return cv2.morphologyEx(
    mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def computeDistTransform(img):
    """
    Compute the distance transform of a binary image
    :param img: binary image
    :return: distance transform
    """
    # assert img.ptp() != 0
    # assert img.max() > 1
    dist = cv2.distanceTransform((img*255).astype(np.uint8), cv2.DIST_L2, 5)
    # normalize the distance transform
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

    return dist


def computeContours(img):
    """
    Compute the contours of a binary image using the Satoshii algorithm.
    :param img: binary image
    :return: contours
    """
    img = (img.copy()*255).astype(np.uint8)

    contours, _ = cv2.findContours(img,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_NONE)
    return contours

##############################################


class Nuclie_data(Dataset):
    """
    Nuclei instance segmentation dataset
    """

    def __init__(self, path):
        """
        Initialize the dataset
        :param path: path to the dataset
        """
        self.path = path
        self.folders = os.listdir(path)
        self.transforms = get_transforms(0.5, 0.5)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        """
        Get the item at the given index
        :param idx: index of the item
        :return: item
        """
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (256, 256))

        (objectProbas, overlapProba, objectContours, mask) = self.get_mask(
            mask_folder, 256, 256)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        # draw a random number between 0 and 1
        if np.random.random() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            overlapProba = overlapProba[:, ::-1]
            objectProbas = objectProbas[:, ::-1]
            objectContours = list(
                map(lambda contour: np.hstack(
                    (256 - contour[:, 0].reshape(-1, 1), contour[:, 1].reshape(-1, 1))), objectContours))

        target = {}
        target["objectProbas"] = objectProbas
        target["overlapProba"] = overlapProba
        target["objectContours"] = objectContours
        target["mask"] = mask

        return img, target

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        """
        Get the mask of a given image from the mask folder of the dataset
        :param mask_folder: path to the mask folder
        :param IMG_HEIGHT: height of the image
        :param IMG_WIDTH: width of the image
        :return: mask
        """

        masks = np.stack([transform.resize(fillHoles(io.imread(os.path.join(mask_folder, mask_))), (IMG_HEIGHT, IMG_WIDTH), 0)
                          for mask_ in os.listdir(mask_folder)], axis=0
                         ).astype(np.float32)  # read all masks

        mask = (np.max(masks, axis=0)*255).astype(np.uint8)

        overlapProba = np.sum(masks, axis=0) > 1.0

        objectProbas = np.max(np.stack(tuple(computeDistTransform(
            masks[i]) for i in range(masks.shape[0])), axis=0), axis=0
        )

        objectContours = [computeContours(masks[i])[0].reshape(-1, 2)
                          for i in range(masks.shape[0])]

        return (objectProbas, overlapProba, objectContours, mask)


def collate_fn(batch):
    """
    Collate function for the dataloader
    :param batch: batch of items
    :return: batch of items
    """
    images, targets, = [], []
    for (img, target) in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


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
        return DataLoader(train_dataset,
                          batch_size=8,
                          prefetch_factor=4,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=1)

    # def val_dataloader(self):
    #     val_dataset = Nuclie_data(
    #         path=os.path.join(os.path.dirname(__file__), "DSB18/test"))
    #     return DataLoader(val_dataset, batch_size=32, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=4, shuffle=False)
