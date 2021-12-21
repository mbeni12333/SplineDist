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
from scipy.interpolate import interp1d
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


def computeContours(img, maxSize=700):
    """
    Compute the contours of a binary image using the Satoshii algorithm.
    :param img: binary image
    :return: contours
    """
    img = (img.copy()*255).astype(np.uint8)

    contours, _ = cv2.findContours(img,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_NONE)

    
    # interpolate contours to have a max size of maxSize

    contours = contours[-1].reshape(-1, 2)

    contourSize = contours.shape[0]

    interpolator_x = interp1d(np.linspace(0, 1, contourSize),
                                contours[:, 0], kind='nearest')
    interpolator_y = interp1d(np.linspace(0, 1, contourSize),
                                contours[:, 1], kind='nearest')

    x = np.linspace(0, 1, num=maxSize)

    return np.array([interpolator_x(x),
                     interpolator_y(x)]).T

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
            objectContours = np.array(list(
                map(lambda contour: np.hstack(
                    (256 - contour[:, 0].reshape(-1, 1), contour[:, 1].reshape(-1, 1))), objectContours)))

        target = {}
        target["objectProbas"] = torch.from_numpy(objectProbas.copy()).float()
        target["overlapProba"] = torch.from_numpy(overlapProba.copy()).float()
        target["objectContours"] = torch.from_numpy(objectContours.copy()).float()
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
        objectContours = np.array([computeContours(masks[i]) for i in range(masks.shape[0])])

        return (objectProbas, overlapProba, objectContours, mask)


def collate_fn(batch):
    """
    Collate function for the dataloader
    :param batch: batch of items
    :return: batch of items
    """
    images = []
    objectProbas, overlapProba, objectContours= [], [], []
    
    for (img, target) in batch:
        images.append(img)
        # targets.append(target)
        objectProbas.append(target["objectProbas"])
        overlapProba.append(target["overlapProba"])
        objectContours.append(target["objectContours"])
        # masks.append(target["mask"])


    images = torch.stack(images, dim=0)
    objectProbas = torch.stack(objectProbas, dim=0).unsqueeze(1)
    # overlapProba = torch.stack(overlapProba, dim=0)
    # objectContours = torch.vstack(objectContours)
    # masks = torch.stack(masks, dim=0)


    return images, (objectProbas, overlapProba, objectContours)
    # return images,targets

class Nuclie_datamodule(pl.LightningDataModule):
    """
    Datamonodule for Nuclei dataset
    """

    def __init__(self, train_len=0.8, val_len=0.2):
        super().__init__()
        dataset = Nuclie_data(path=os.path.join(os.path.dirname(__file__),
                                "DSB18/train"))

        self.trainDataset, self.valDataset = random_split(dataset,[int(train_len*len(dataset)), int(val_len*len(dataset))])


    def train_dataloader(self):
        """
        Get the train dataloader
        :return: train dataloader
        """
        return DataLoader(self.trainDataset,
                          batch_size=8,
                          prefetch_factor=4,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=8)

    def val_dataloader(self):
        """
        Get the validation dataloader
        :return: validation dataloader
        """
        return DataLoader(self.valDataset,
                          batch_size=4,
                          prefetch_factor=4,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=collate_fn,
                          num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=4, shuffle=False)
