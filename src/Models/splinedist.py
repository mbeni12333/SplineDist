import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from nms import nms, helpers
# from pytorch_lightning.metrics.functional import accuracy, mAP
# from scipy.interpolate import BSpline, splev
# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
from torch import cdist
from .unet import UNet
from utils import *
import tqdm



###########################################################################
def linear_sum_assignment(cost_matrix):
    """Solve the linear sum assignment problem.

    The linear sum assignment problem is also known as minimum weight matching
    in bipartite graphs. A problem instance is described by a matrix C, where
    each C[i,j] is the cost of matching vertex i of the first partite set
    (a "worker") and vertex j of the second set (a "job"). The goal is to find
    a complete assignment of workers to jobs of minimal cost.

    Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is
    assigned to column j. Then the optimal assignment has cost

    .. math::
        \min \sum_i \sum_j C_{i,j} X_{i,j}

    s.t. each row is assignment to at most one column, and each column to at
    most one row.

    This function can also solve a generalization of the classic assignment
    problem where the cost matrix is rectangular. If it has more rows than
    columns, then not every row needs to be assigned to a column, and vice
    versa.

    The method used is the Hungarian algorithm, also known as the Munkres or
    Kuhn-Munkres algorithm.

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.

    Notes
    -----
    .. versionadded:: 0.17.0

    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> from scipy.optimize import linear_sum_assignment
    >>> row_ind, col_ind = linear_sum_assignment(cost)
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind].sum()
    5

    References
    ----------
    1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       *Naval Research Logistics Quarterly*, 2:83-97, 1955.

    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       *J. SIAM*, 5(1):32-38, March, 1957.

    5. https://en.wikipedia.org/wiki/Hungarian_algorithm
    """

    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _Hungary(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    if transposed:
        marked = state.marked.T
    else:
        marked = state.marked
    return np.where(marked == 1)


class _Hungary(object):
    """State of the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Must have shape[1] >= shape[0].
    """

    def __init__(self, cost_matrix):
        self.C = cost_matrix

        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=bool)
        self.col_uncovered = np.ones(m, dtype=bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= np.asarray(state.col_uncovered, dtype=int)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    np.asarray(state.row_uncovered, dtype=int))
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4

####################################################################################################

from concurrent.futures import ProcessPoolExecutor

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

MAX_CONTOUR_SIZE = 500

################################# Helper Functions ################################################################


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
    cost = cdist(contour1.detach(),
                 contour2.detach())
    # then we solve the linear affecteion problem
    rowId, colId = linear_sum_assignment(cost.cpu().numpy())
    # we obtain the distance between the contour pixels

    dist = torch.abs(contour1[rowId] - contour2[colId]).sum()
    return dist
    # return cost[rowId, colId].sum()

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

def polygon_iou(poly1, poly2, useCV2=True):
    """Computes the ratio of the intersection area of the input polygons to the (sum of polygon areas - intersection area)
    Used with the NMS function

    :param poly1: a polygon described by its verticies
    :type poly1: list
    :param poly2: a polygon describe by it verticies
    :type poly2: list
    :type useCV2: bool
    :return: The ratio of the intersection area / (sum of rectangle areas - intersection area)
    :rtype: float
    """
    poly1 = poly1.reshape(MAX_CONTOUR_SIZE, 2)
    poly2 = poly2.reshape(MAX_CONTOUR_SIZE, 2)
    
    intersection_area = helpers.polygon_intersection_area([poly1, poly2])
    if intersection_area == 0:
        return 0


    poly1_area = cv2.contourArea(np.array(poly1, np.int32))
    poly2_area = cv2.contourArea(np.array(poly2, np.int32))

    try:
        iou = intersection_area / (poly1_area + poly2_area - intersection_area)
        return iou
    except ZeroDivisionError:
        return 0


def computeContourLoss(objectProbas, contours, targetObjectProbas, targetContours, nms_threshold=0.7, obj_threshold=0.7):
    """
    """
    
    scores = objectProbas.flatten(1).cpu().detach().numpy()
    sortedInstances = np.argsort(scores, axis=1).copy()
    
    til = []
    for i in range(sortedInstances.shape[0]):
        til.append(np.searchsorted(scores[i, sortedInstances[i]], obj_threshold))

    contours2 = [contours[i, til[i]:].detach().cpu().numpy() for i in range(len(til))]

    loss = 0

    for i in range(targetObjectProbas.shape[0]):
        term1 = 0
        term2 = 0

        postProcessed = nonMaximumSuppresion(scores[i, sortedInstances[i, til[i]:]],
                                            contours2[i],
                                            obj_threshold,
                                            nms_threshold)

        contoursSelected, idsSelected = postProcessed
        idsSelected = np.array(idsSelected)

        cost = np.zeros((len(contoursSelected), len(targetContours[i])))
        for j in range(len(idsSelected)):
            for k in range(len(targetContours[i])):
                cost[j, k] = polygon_iou(contoursSelected[j].reshape(MAX_CONTOUR_SIZE*2),
                                         targetContours[i][k].reshape(MAX_CONTOUR_SIZE*2).cpu().numpy())

        rowId, colId = linear_sum_assignment(-cost)

        rowId2 = rowId[cost[rowId, colId] > 0]
        colId2 = colId[cost[rowId, colId] > 0]

        idsSelected2 = idsSelected[rowId2]
        TP = [contours[i, idsSelected2], idsSelected2, colId2]

        if len(idsSelected2) > 0:
            term1 = torch.stack(tuple(computeDistanceBetweenInstance(TP[0][j],
                                      targetContours[i][TP[2]][j]) for j in range(len(idsSelected2)))).mean()
            term2 = torch.abs(contours[i, ~idsSelected2]).mean()
        else:
            term2 = torch.abs(contours[i, :]).mean()

        loss += term1 + term2

    return loss/targetObjectProbas.shape[0]


def nonMaximumSuppresion(objectProbas, contours, score_threshold=0.8, iou_threshold=0.7):
    """
    This function performs non maximum suppression on a single contour
    :param objectProbas: the object probabilities
    :param contours: the contours
    :param score_threshold: the score threshold to use
    :param iou_threshold: the IoU threshold to use
    :return: the non maximum suppressed contour
    """

    ids = nms.fast.nms(contours,
                       objectProbas,
                       score_threshold=score_threshold,
                       iou_threshold=iou_threshold,
                       compare_function=polygon_iou)

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

        for future in tqdm.tqdm(futures):
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

        self.B3M = getBsplineMatrix(numSamples=MAX_CONTOUR_SIZE,
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


        # targetContours = list(map(lambda x: x.to(device), targetContours))
        # targetObjectProbas = targetObjectProbas.to(device)

        contours = getContourSamples(controlPoints, self.B3M)
        contours = contours.permute(1, 0, 2, 3, 4).reshape(objectProbas.shape[0], -1, MAX_CONTOUR_SIZE, 2)
        # loss 1
        lossObjectProba = F.binary_cross_entropy(objectProbas, targetObjectProbas)
        lossContour = computeContourLoss(objectProbas, contours, targetObjectProbas, targetContours)
        # # loss 2
        # # non maximum suppression
        # postProcessed = nonMaximumSuppresionBatch(objectProbas,
        #                                           controlPoints,
        #                                           self.hparams["object_threshold"],
        #                                           self.hparams["nms_threshold"])
        # # find the associated instances from postProcessed with the ground truth
        # TP, FP, FN = matchInstances(postProcessed, targetContours)
        # torch.stack(tuple(computeDistanceBetweenInstance(postProcessed[id1], targetContours[id2]) 
        #                         for id1, id2 in TP)).sum()
        # precision = len(TP) / (len(TP) + len(FP) + 1)
        # recall = len(TP) / (len(TP) + len(FN)+1)
        # # compute the sum of the distances with predicted instances and the ground truth
        # lossDistance = 0
        # loss 3 sum of the two losses
        finalLoss = lossObjectProba + 0.1*lossContour

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
