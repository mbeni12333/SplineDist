import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
# from pytorch_lightning.metrics.functional import accuracy, mAP

from .unet import UNet

# create a pytorch lightning module that uses unet model


class SplineDist(pl.LightningModule):
    """
    This model is not working yet.
    """

    def __init__(self,
                 num_control_points=8,
                 num_classes=1,
                 learning_rate=1e-3,
                 nms_threshold=0.5,
                 object_threshold=0.5,
                 num_features=128,
                 discretization=64,
                 splineBasis=3):
        """
        """
        super().__init__()
        # self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 3, 256, 256)

        self.segmentationBackbone = UNet()

        self.objectFinder = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding="same"),
                                          nn.Sigmoid())

        self.ControlPointsAngleRegressor = nn.Conv2d(
            128, num_control_points, kernel_size=3, padding="same")
        self.ControlPointsDistanceRegressor = nn.Conv2d(
            128, num_control_points, kernel_size=3, padding="same")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("splinedist")

        parser.add_argument("--num_control_points", type=int, default=8)
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--nms_threshold", type=float, default=0.5)
        parser.add_argument("--object_threshold", type=float, default=0.5)
        parser.add_argument("--num_features", type=int, default=128)
        parser.add_argument("--discretization", type=int, default=64)
        parser.add_argument("--splineBasis", type=int, default=3)

        return parent_parser

    def forward(self, x):

        features = self.segmentationBackbone(x)
        objectProbas = self.objectFinder(features)
        angles = self.ControlPointsAngleRegressor(features)
        distances = self.ControlPointsDistanceRegressor(features)

        return objectProbas, angles, distances

    def coordinatesToInstances(self, coordinates):
        """
        """
        pass

    def discretizeBspline(self, angles, distances):
        """
        """
        pass

    def matchInstances():
        """
        """
        pass

    def nms(self, boxes, scores, threshold=0.5):
        """
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def compute_loss(self, ):
        """
        """
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        objectProbas, angles, distances = self(x)
        loss = self.compute_loss(objectProbas, angles, distances, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        objectProbas, angles, distances = self(x)
        loss = self.compute_loss(objectProbas, angles, distances, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def configure_optimizers(self):
        """
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]
