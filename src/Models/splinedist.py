import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from nms import nms
# from pytorch_lightning.metrics.functional import accuracy, mAP
from scipy.interpolate import BSpline, splev
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .unet import UNet
from utils import *

from concurrent.futures import ProcessPoolExecutor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################# Helper Functions ################################################################


# def computeSplineSamples(controlPoints, numSamples, degree=3):
#     """
#     This function computes the spline of given control points discretization
#     :param controlPoints: the control points of the spline
#     :param numSamples: the number of samples to compute
#     :param degree: the degree of the spline
#     :return: the spline samples
#     """
#     # return BSpline(np.linspace(0, 1, degree+controlPoints.shape[0]+1), controlPoints, degree)
#     # If periodic, extend the point array by count+degree+1
#     count = len(controlPoints)
#     # factor, fraction = divmod(count+degree+1, count)

#     # controlPoints = torch.concat((controlPoints,) * factor + (controlPoints[:fraction],))
#     # count = len(controlPoints)

#     # degree = np.clip(degree, 1, degree)

#     # Calculate knot vector
#     kv = np.arange(0-degree, count+degree+degree-1, dtype='int')

#     # Calculate query range
#     u = np.linspace(1, (count-degree), numSamples)

#     # Calculate result
#     return splev(u, (kv, controlPoints.T, degree))


def matchInstances(postProcessed, target):

    cost = cdist(postProcessed, target, metric=nms.fast.polygon_iou)

    rowId, colId = linear_sum_assignment(cost)



    TP = []
    FP = []
    FN = []

    return TP, FP, FN


def computeDistanceBetweenInstance(contour1, contour2):
    """
    This function computes the distance between two instances
    :param contour1: the first contour
    :param contour2: the second contour
    :return: the distance between the two instances
    """
    # we must match every pixel of the contour to the corresponding pixel of the other contour
    # we use the euclidean distance to compute the distance between the contour pixels
    cost = cdist(contour1, contour2, metric="euclidean")
    # then we solve the linear affecteion problem
    rowId, colId = linear_sum_assignment(cost)
    # we obtain the distance between the contour pixels
    return cost[rowId, colId].sum()

#################### TO REMOVE ################################
def convertOutputsToControlPoints(angles, distances, size=256):
    """
    """
    controlPoints = torch.stack((torch.cos(angles)*distances,
                                 torch.sin(angles)*distances), dim=2)
    x = torch.arange(size)
    y = torch.arange(size)
    xx, yy = torch.meshgrid(x, y)
    shifts = torch.stack((xx, yy), dim=0)

    controlPoints = controlPoints + shifts

    return controlPoints
#################### TO REMOVE ################################

def nonMaximumSuppresion(objectProbas, contours, score_threshold=0.8, iou_threshold=0.7):
    """
    This function performs non maximum suppression on a single contour
    :param objectProbas: the object probabilities
    :param contours: the contours
    :param score_threshold: the score threshold to use
    :param iou_threshold: the IoU threshold to use
    :return: the non maximum suppressed contour
    """

    ids = nms.polygons(contours,
                       objectProbas,
                       score_threshold=score_threshold,
                       iou_threshold=iou_threshold)

    # ids = np.unravel_index(ids, contours.shape, order='C')

    return contours[ids], ids


def nonMaximumSuppresionBatch(objectProbas, contours, score_threshold=0.8, iou_threshold=0.7):
    """
    This function performs non maximum suppression on a batch of contours
    :param objectProbas: the object probabilities
    :param controlPoints: the control points of the contours
    :param threshold: the threshold to use
    :return: the non maximum suppressed contours
    """
    postProcessed = []
    # for i in prange(objectProbas.shape[0]):
    #     postProcessed.append(nonMaximumSuppresion(objectProbas[i], contours[i], score_threshold, iou_threshold)

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(objectProbas.shape[0]):
            futures.append(executor.submit(nonMaximumSuppresion, objectProbas[i], contours[i], score_threshold, iou_threshold))

        for future in futures:
            postProcessed.append(future.result()) # this will block

    # postProcessed = [nonMaximumSuppresion(objectProbas[i], contours[i], score_threshold, iou_threshold)
    #                                             for i in range(objectProbas.shape[0])]

    return postProcessed
################################# Model Class ###################################################################

class Cos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.cos(input)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


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
                 lambda2=0.1,
                 size=256):
        """
        """
        # initialize the parent class
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 3, 256, 256)

        # Modules of the model
        self.segmentationBackbone = UNet()
        self.objectFinder = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, padding=1),
                                          nn.Sigmoid())

        self.ControlPointsAngleRegressor = nn.Conv2d(
            num_features, num_control_points, kernel_size=3, padding=1)

        self.ControlPointsDistanceRegressor = nn.Conv2d(
            num_features, num_control_points, kernel_size=3, padding=1)

        x = torch.arange(size)
        y = torch.arange(size)
        xx, yy = torch.meshgrid(x, y)
        self.shifts = torch.stack((xx, yy), dim=0).to(device)
        self.cos = Cos()
        self.sin = Sin()

        self.B3M = getBsplineMatrix(numSamples=700,
                                    degree=splineBasis,
                                    numControlPoints=8).float().to(device)

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
        controlPoints = torch.stack((self.cos(angles)*distances,
                                     self.sin(angles)*distances), dim=2) + self.shifts

        return objectProbas, angles, distances, controlPoints


    def matchInstances():
        """
        """
        pass

    def compute_loss(self, predicted_targets, targets):
        """
        This function computes the loss of the model
        :param predicted_targets: the predicted targets
        :param targets: the targets
        """
        (objectProbas, controlPoints) = predicted_targets
        (targetObjectProbas, targetOverlapProbas, targetContours) = targets

        contours = getContourSamples(controlPoints, self.B3M)
        # loss 1
        lossObjectProba = F.binary_cross_entropy(objectProbas, targetObjectProbas)

        # loss 2
        # non maximum suppression
        postProcessed = nonMaximumSuppresionBatch(objectProbas,
                                                  controlPoints,
                                                  self.hparams["object_threshold"],
                                                  self.hparams["nms_threshold"])
        # find the associated instances from postProcessed with the ground truth
        TP, FP, FN = matchInstances(postProcessed, targetContours)
        torch.stack(tuple(computeDistanceBetweenInstance(postProcessed[id1], targetContours[id2]) 
                                for id1, id2 in TP)).sum()
        precision = len(TP) / (len(TP) + len(FP) + 1)
        recall = len(TP) / (len(TP) + len(FN)+1)
        # compute the sum of the distances with predicted instances and the ground truth
        lossDistance = 0
        # loss 3 sum of the two losses
        finalLoss = lossObjectProba + self.lambda1*lossDistance

        return finalLoss

    def training_step(self, batch, batch_idx):
        """
        This function is called for each batch
        :param batch: the batch
        :param batch_idx: the batch index
        """
        images, targets = batch
        objectProbas, angles, distances, controlPoints = self(images)
        predicted_targets = (objectProbas, controlPoints)
        loss = self.compute_loss(predicted_targets, targets)
        self.log("train_loss", loss)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):

        images, targets = batch
        objectProbas, angles, distances, controlPoints = self(images)
        predicted_targets = (objectProbas, controlPoints)

        loss = self.compute_loss(predicted_targets, targets)
        self.log("val_loss", loss)
        self.log("hp_metric", loss)

        return {"loss": loss}


    def configure_optimizers(self):
        """

        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]
