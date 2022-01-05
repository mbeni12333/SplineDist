import os
from Models.splinedist import MAX_CONTOUR_SIZE
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys


from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, utils, io
#from skimage import io as skio
from torchvision.transforms import (RandomHorizontalFlip, Normalize, Resize, Compose, ToTensor)

import cv2
from torch import nn
import pytorch_lightning as pl

from numpy import interp


MAX_CONTOUR_SIZE = 291

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transforms(mean, std):
    """
    Define the transformations to be used for the dataset
    :param mean: mean of the dataset
    :param std: standard deviation of the dataset
    :return: transformations
    """
    list_transforms = [
        Resize((256, 256)),
        Normalize(mean=mean, std=std),        
    ]

    return Compose(list_transforms)

############## Helper Functions ##############


def fillHoles(img):
#     return cv2.morphologyEx(
#         img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
#     )
    return  img

#     th, im_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV);
#     im_floodfill = im_th.copy()
    
#     h, w = img.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
#     # Floodfill from point (0, 0)
#     cv2.floodFill(img, mask, (0,0), 255);
#     # Invert floodfilled image
#     im_floodfill_inv = cv2.bitwise_not(im_floodfill)


#     return im_floodfill_inv


def computeDistTransform(img):
    """
    Compute the distance transform of a binary image
    :param img: binary image
    :return: distance transform
    """
    # assert img.ptp() != 0
    # assert img.max() > 1

    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    # normalize the distance transform
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

    return dist


def computeContours(img, maxSize=700):
    """
    Compute the contours of a binary image using the Satoshii algorithm.
    :param img: binary image
    :return: contours
    """
    if img.sum() == 0:
        return []
    
    contours, _ = cv2.findContours(img,
                                   mode=cv2.RETR_EXTERNAL ,
                                   method=cv2.CHAIN_APPROX_NONE)  


    contours = contours[0].reshape(-1, 2)

    contourSize = contours.shape[0]
    
    interpolator_x = interp(np.linspace(0, 1, maxSize),
                              np.linspace(0, 1, contourSize), contours[:, 0])
    interpolator_y = interp(np.linspace(0, 1, maxSize),
                              np.linspace(0, 1, contourSize), contours[:, 1])
    
    return np.array([interpolator_x, interpolator_y]).T

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

        img = io.read_image(image_path, io.ImageReadMode.RGB).float()/255

        # img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
        # img = torch.from_numpy(img).permute(2, 0, 1).float()

        (objectProbas, overlapProba, objectContours, mask) = self.get_mask(
            mask_folder, 256, 256)

        img = self.transforms(img)

#         # Horizontal flip
#         if np.random.random() > 0.5:
#             img = transforms.functional.hflip(img)
#             mask = mask[:, ::-1]
#             overlapProba = overlapProba[:, ::-1]
#             objectProbas = objectProbas[:, ::-1]
#             m = objectContours == 0
#             objectContours[:, :, :, 0] = 255 - objectContours[:, :, :, 0]
#             objectContours = objectContours*m
#             # objectContours = list(
#             #     map(lambda contour: np.hstack(
#             #         (255 - contour[:, 0].reshape(-1, 1), contour[:, 1].reshape(-1, 1))), objectContours))

#         # Vertical flip
#         if np.random.random() > 0.5:
#             img = transforms.functional.vflip(img)
#             mask = mask[::-1, :]
#             overlapProba = overlapProba[::-1, :]
#             objectProbas = objectProbas[::-1, :]
#             m = objectContours == 0
#             objectContours[:, :, :, 1] = 255 - objectContours[:, :, :, 1]
#             objectContours = objectContours*m

        target = {}
        target["objectProbas"] = torch.from_numpy(objectProbas.copy()).float()
        target["overlapProba"] = torch.from_numpy(overlapProba.copy()).float()
        target["objectContours"] = torch.from_numpy(objectContours.copy()).float()
        # target["objectContours"] = objectContours
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
  

        masks = np.stack([
            cv2.resize(
                    io.read_image(os.path.join(mask_folder, mask_ ), io.ImageReadMode.GRAY).permute(1, 2, 0).numpy(),
                (IMG_HEIGHT, IMG_WIDTH)
            )
            for mask_ in os.listdir(mask_folder)], axis=0
        )

        mask = np.max(masks, axis=0)

        overlapProba = np.sum(masks//255, axis=0) > 1.0
        
        objectProbas = []
        # objectContours = []

        objectContours = np.zeros((IMG_HEIGHT, IMG_WIDTH, MAX_CONTOUR_SIZE, 2), dtype=np.uint8)

        for i in range(masks.shape[0]):
            objectProbas.append(computeDistTransform(masks[i]))
            contour = computeContours(masks[i], MAX_CONTOUR_SIZE)
            idx_nonzero  = np.argwhere(masks[i])

            if contour != []:
                # objectContours.append(contour)
                roll = contour[:, 1].argmin()
                contour = np.roll(contour, roll, 0)
                objectContours[idx_nonzero[:, 0], idx_nonzero[:, 1]] = contour
            
        objectProbas = np.stack(objectProbas, axis=0).max(0)
        # objectContours = np.array(objectContours)

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
    objectContours = torch.stack(objectContours, dim=0)
    objectProbas = torch.stack(objectProbas, dim=0).unsqueeze(1)
    overlapProba = torch.stack(overlapProba, dim=0).unsqueeze(1)

    return images, (objectProbas, overlapProba, objectContours)


class Nuclie_datamodule(pl.LightningDataModule):
    """
    Datamonodule for Nuclei dataset
    """

    def __init__(self, train_len=0.8, val_len=0.2, path=""):
        super().__init__()
        dataset = Nuclie_data(path=path)

        self.trainDataset, self.valDataset = random_split(dataset,[int(train_len*len(dataset)), int(val_len*len(dataset))])


    def train_dataloader(self):
        """
        Get the train dataloader
        :return: train dataloader
        """
        return DataLoader(self.trainDataset,
                          batch_size=4,
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
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=collate_fn,
                          num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=4, shuffle=False)
