import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
# from pytorch_lightning.metrics.functional import accuracy, mAP
from scipy.interpolate import BSpline, splev
from scipy.optimize import linear_sum_assignment
from .unet import UNet

################################# Helper Functions ################################################################


def computeSplineSamples(controlPoints, numSamples, degree=3):
    """
    This function computes the spline of given control points discretization
    :param controlPoints: the control points of the spline
    :param numSamples: the number of samples to compute
    :param degree: the degree of the spline
    :return: the spline samples
    """
    # return BSpline(np.linspace(0, 1, degree+controlPoints.shape[0]+1), controlPoints, degree)
    # If periodic, extend the point array by count+degree+1
    count = len(controlPoints)
    # factor, fraction = divmod(count+degree+1, count)

    # controlPoints = torch.concat((controlPoints,) * factor + (controlPoints[:fraction],))
    # count = len(controlPoints)

    # degree = np.clip(degree, 1, degree)

    # Calculate knot vector
    kv = np.arange(0-degree, count+degree+degree-1, dtype='int')

    # Calculate query range
    u = np.linspace(1, (count-degree), numSamples)

    # Calculate result
    return splev(u, (kv, controlPoints.T, degree))


def computeDistanceBetweenInstance(contour1, contour2):
    """
    This function computes the distance between two instances
    :param contour1: the first contour
    :param contour2: the second contour
    :return: the distance between the two instances
    """
    # we must match every pixel of the contour to the corresponding pixel of the other contour
    # we use the euclidean distance to compute the distance between the contour pixels
    cost = cdist(contour1, contour2, metric='euclidean')
    # then we solve the linear affecteion problem
    rowId, colId = linear_sum_assignment(cost)
    # we obtain the distance between the contour pixels
    return cost[row_ind, col_ind].sum()

################################# Model Class ###################################################################


class SplineDist(pl.LightningModule):
    """
    This model predicts an object instance for each pixel of the image
    """

    def __init__(self,
                 num_control_points=8,
                 num_classes=1,
                 learning_rate=1e-3,
                 nms_threshold=0.5,
                 object_threshold=0.5,
                 num_features=128,
                 splineBasis=3,
                 lambda1=1,
                 lambda2=0.1):
        """
        """
        # initialize the parent class
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 3, 256, 256)

        # Modules of the model
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
        parser.add_argument("--splineBasis", type=int, default=3)
        parser.add_argument("--lambda1", type=float, default=1)
        parser.add_argument("--lambda2", type=float, default=0.1)

        return parent_parser

    def forward(self, x):

        features = self.segmentationBackbone(x)
        objectProbas = self.objectFinder(features)
        angles = self.ControlPointsAngleRegressor(features)
        distances = self.ControlPointsDistanceRegressor(features)

        return objectProbas, angles, distances

    def coordinatesToInstances(self, coordinates):
        """
        This function transforms the coordinates of the contour to Object instances
        :param coordinates: the coordinates of the contour
        :return: the Object instances
        """
        pass

    def matchInstances():
        """
        """
        pass

    # def nms(self, coordinates, scores, threshold=0.5):
    #     """
    #     """
    #     sortedInstances = scores.argsort()[::-1]

    #     keep = []
    #     while sortedInstances.size > 0:
    #         # keep the highest scoring instance
    #         currentInstance = sortedInstances.pop()
    #         keep.append(currentInstance)

    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])

    #         w = np.maximum(0.0, xx2 - xx1)
    #         h = np.maximum(0.0, yy2 - yy1)
    #         inter = w * h
    #         ovr = inter / (areas[i] + areas[order[1:]] - inter)

    #         inds = np.where(ovr <= threshold)[0]
    #         order = order[inds + 1]

    #     return keep

    def compute_loss(self, predicted_targets, targets):
        """
        This function computes the loss of the model
        :param predicted_targets: the predicted targets
        :param targets: the targets
        """
        return torch.tensor([0.0], requires_grad=True)

    def training_step(self, batch, batch_idx):
        """
        This function is called for each batch
        :param batch: the batch
        :param batch_idx: the batch index
        """
        images, targets = batch
        objectProbas, angles, distances = self(images)
        predicted_targets = (objectProbas, angles, distances)
        loss = self.compute_loss(predicted_targets, targets)

        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     objectProbas, angles, distances = self(x)
    #     loss = self.compute_loss(objectProbas, angles, distances, y)
    #     return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     return {"val_loss": avg_loss}

    def configure_optimizers(self):
        """

        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]
