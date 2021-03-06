import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from nms import nms, helpers
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, batched_nms
import matplotlib.patches as patches
from torch import cdist
from .unet import UNet
from utils import *
import tqdm
import pdb
from torchvision.utils import make_grid
import matplotlib as mpl
from shapely.geometry import Polygon
from shapely.validation import make_valid
import rasterio.features

################################ Remove #################################

###########################################################################
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

MAX_CONTOUR_SIZE = 291

colors = [ list(map(lambda x: x/255, getRandomColor())) for i in range(256*256+1)]
################################# Helper Functions ################################################################




def computeDistanceBetweenInstance(contour1, contour2):
    """
    This function computes the distance between two instances
    :param contour1: the first contour
    :param contour2: the second contour
    :return: the distance between the two instances
    """
    # we must match every pixel of the contour to the corresponding pixel of the other contour
    # we use the euclidean distance to compute the distance between the contour pixels
    with torch.no_grad():
        cost = cdist(contour1,
                     contour2)
        # then we solve the linear affecteion problem

#         rowId, colId = linear_sum_assignment(cost.cpu().numpy())
    # we obtain the distance between the contour pixels
    m = cost.argmin().item()
    i, j = m//291, m%291
    dist = torch.abs(torch.roll(contour1, (j-i+1), 0) - contour2).mean()
#     dist = torch.abs(contour1[rowId] - contour2[colId]).sum()

    return dist
    # return cost[rowId, colId].sum()

# #################### TO REMOVE ################################

# #################### TO REMOVE ################################
def createImage(width=800, height=800, depth=3):
    """ Return a black image with an optional scale on the edge

    :param width: width of the returned image
    :type width: int
    :param height: height of the returned image
    :type height: int
    :param depth: either 3 (rgb/bgr) or 1 (mono).  If 1, no scale is drawn
    :type depth: int
    :return: A zero'd out matrix/black image of size (width, height)
    :rtype: :class:`numpy.ndarray`
    """
    # create a black image and put a scale on the edge

    assert depth == 3 or depth == 1
    assert width > 0
    assert height > 0

    hashDistance = 50
    hashLength = 20

    img = np.zeros((int(height), int(width), depth), np.uint8)

    if(depth == 3):
        for x in range(0, int(width / hashDistance)):
            cv2.line(img, (x * hashDistance, 0), (x * hashDistance, hashLength), (0,0,255), 1)

        for y in range(0, int(width / hashDistance)):
            cv2.line(img, (0, y * hashDistance), (hashLength, y * hashDistance), (0,0,255), 1)

    return img



def IOU2(pol1_xy, pol2_xy, over=None):
    # Define each polygon
    
    poly1 = pol1_xy.reshape(-1, 2)
    poly2 = pol2_xy.reshape(-1, 2)
    
    polygon1_shape = make_valid(Polygon(pol1_xy))
    polygon2_shape = make_valid(Polygon(pol2_xy))
#     polygon1_shape = Polygon(pol1_xy)
#     polygon2_shape = Polygon(pol2_xy)
#     print(polygon1_shape.is_valid, polygon2_shape.is_valid)

    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape)
    if over is not None and polygon_intersection.area != 0:
        polygon_intersection = rasterio.features.rasterize([polygon_intersection], out_shape=(256, 256))
        polygon_intersection = over[polygon_intersection>0]
        polygon_intersection = (1 - polygon_intersection).sum()
    else:
        polygon_intersection = polygon_intersection.area
#         print("Hello")

    
#     print(polygon_intersection)
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
#     print(polygon_intersection, polygon_union)
    try:
        IoU = polygon_intersection / polygon_union
        return IoU
    except ZeroDivisionError:
        return 0

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
    poly1 = poly1.reshape(-1, 2)
    poly2 = poly2.reshape(-1, 2)
    

    
    intersection_area = polygon_intersection_area([poly1, poly2])
    if intersection_area == 0:
        return 0

    poly1_area = cv2.contourArea(np.array(poly1, np.int32))
    poly2_area = cv2.contourArea(np.array(poly2, np.int32))
    
    try:
        iou = intersection_area / (poly1_area + poly2_area - intersection_area)
#         print(poly1_area, poly2_area, intersection_area, iou)
        return iou
    except ZeroDivisionError:
        return 0


    
    
def polygon_intersection_area(polygons):
    """ Compute the area of intersection of an array of polygons

    :param polygons: a list of polygons
    :type polygons: list
    :return: the area of intersection of the polygons
    :rtype: int
    """
    if len(polygons) == 0:
        return 0

    dx = 0
    dy = 0

    maxx = np.amax(np.array(polygons)[...,0])
    minx = np.amin(np.array(polygons)[...,0])
    maxy = np.amax(np.array(polygons)[...,1])
    miny = np.amin(np.array(polygons)[...,1])

    if minx < 0:
        dx = -int(minx)
        maxx = maxx + dx
    if miny < 0:
        dy = -int(miny)
        maxy = maxy + dy
    # (dx, dy) is used as an offset in fillPoly

    for i, polypoints in enumerate(polygons):

        newImage = createImage(maxx, maxy, 1)

        polypoints = np.array(polypoints, np.int32)
        polypoints = polypoints.reshape(-1, 1, 2)

        cv2.fillPoly(newImage, [polypoints], (255, 255, 255), cv2.LINE_8, 0, (dx, dy))

        if(i == 0):
            compositeImage = newImage
        else:
            compositeImage = cv2.bitwise_and(compositeImage, newImage)

        area = cv2.countNonZero(compositeImage)

    return area

def computeContourLoss(objectProbas, contours, targetObjectProbas, targetOverlap, targetContours, nms_threshold=0.7, obj_threshold=0.7, lambda2=1.0, shifts=None):
    """
    """
#     print(targetObjectProbas.shape, targetOverlap.shape)
    targetObjectProbas = targetObjectProbas.squeeze(1)
    mask_proba = (targetObjectProbas != 0)
    mask = mask_proba * (targetOverlap < 0.5)
    
    mask_norm = mask.flatten(1).sum(1).reshape(-1, 1, 1)
    N = targetObjectProbas.shape[0]

    contour_term = torch.sqrt(((contours - targetContours)**2).mean((-1, -2)))
#     contour_term = torch.abs(contours - targetContours).mean((-1, -2))
    contour_zero = torch.sqrt(((contours - shifts)**2).mean((-1, -2)))
    contour_zero = torch.abs(contours - shifts).mean((-1, -2))
    
    term1 = (targetObjectProbas * mask * contour_term/mask_norm).sum()/N
    term2 = ((~mask_proba) * contour_zero/(256*256-mask_norm)).sum()/N
    
    loss = term1 + lambda2*term2


    return loss


# def computeContourLoss(objectProbas, contours, targetObjectProbas, targetContours, nms_threshold=0.7, obj_threshold=0.7, lambda2=1.0, shifts=None):
#     """
#     """

#     xmin = torch.amin(contours[:, :, :, 1], dim=-1) 
#     xmax = torch.amax(contours[:, :, :, 1], dim=-1)
#     ymin = torch.amin(contours[:, :, :, 0], dim=-1)
#     ymax = torch.amax(contours[:, :, :, 0], dim=-1)

#     bboxes = torch.stack([xmin, ymin, xmax, ymax], -1)
    
#     targetObjectProbas = targetObjectProbas.to(contours.device).reshape(-1, 256*256)
    


#     loss = torch.FloatTensor([[0]]).to(contours.device)
#     for i in range(targetObjectProbas.shape[0]):
#         term1 = 0
#         term2 = 0
#         term1Bis = 0
#         # get the ground truth scores
#         scores = targetObjectProbas[i]

#         # select contours with a ground truth score above 0 
#         contoursWithNotNullProba = contours[i][scores > 0]
#         # then match it to the predicted contours suing intersection over union

#         xmin, _ = targetContours[i][:, :, 1].min(-1) 
#         xmax, _ = targetContours[i][:, :, 1].max(-1) 
#         ymin, _ = targetContours[i][:, :, 0].min(-1) 
#         ymax, _ = targetContours[i][:, :, 0].max(-1)

#         targetBboxes = torch.stack([xmin, ymin, xmax, ymax], -1).to(contours.device)

#         cost = box_iou(bboxes[i][scores>0], targetBboxes)

#         values, selectedIds = cost.max(1)

#         selectedContoursHavingIouNotNull = contours[i][scores>0][values != 0]
#         associatedTargetContours = targetContours[i][selectedIds][values!= 0].to(contours.device)
#         notSelectedContours = (contours[i] - shifts.permute(1, 2, 0).reshape(-1, 1, 2))[scores==0]
        

#         w1 = scores[scores>0][values != 0].reshape(-1, 1, 1)
        

#         # cost = cdist(selectedContoursHavingIouNotNull, associatedTargetContours).flatten(1).detach().cpu().numpy()
#         # m = cost.argmin(1)
#         # i, j = m//291, m%291
#         # roll = tuple((j-i+1))
#         # dist = 0
#         # for k in range(selectedContoursHavingIouNotNull.shape[0]):
#         #     dist += torch.abs(torch.roll(selectedContoursHavingIouNotNull[k], roll[k], 0) - associatedTargetContours[k]).mean()*w1[k]
#         # term1 = dist
        

#         term2 = (lambda2*torch.abs(notSelectedContours)).mean()
        
#         if len(selectedContoursHavingIouNotNull) != 0:
#             term1 = (torch.abs(selectedContoursHavingIouNotNull - associatedTargetContours).mean((1, 2))*w1).mean()
#             # term1 = term1/selectedContoursHavingIouNotNull.shape[0]
#             loss += term1

#         loss += term2


#     return loss


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """ Get the max scores with corresponding indicies

    Adapted from the OpenCV c++ source in `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L33>`__

    :param scores: a list of scores
    :type scores: list
    :param threshold: consider scores higher than this threshold
    :type threshold: float
    :param top_k: return at most top_k scores; if 0, keep all
    :type top_k: int
    :param descending: if True, list is returened in descending order, else ascending
    :returns: a  sorted by score list  of [score, index]
    """
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    # Sort the score pair according to the scores in descending order
    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:,0].argsort()[::-1]] #descending order
    else:
        npscores = npscores[npscores[:,0].argsort()] # ascending order

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()
def nms(boxes, scores, **kwargs):
    """Do Non Maximal Suppression

    As translated from the OpenCV c++ source in
    `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L67>`__
    which was in turn inspired by `Piotr Dollar's NMS implementation in EdgeBox. <https://goo.gl/jV3JYS>`_

    This function is not usually called directly.  Instead use :func:`nms.nms.boxes`, :func:`nms.nms.rboxes`,
    or :func:`nms.nms.polygons`

    :param boxes:  the boxes to compare, the structure of the boxes must be compatible with the compare_function.
    :type boxes:  list
    :param scores: the scores associated with boxes
    :type scores: list
    :param kwargs: optional keyword parameters
    :type kwargs: dict (see below)
    :returns: an list of indicies of the best boxes
    :rtype: list
    :kwargs:

    * score_threshold (float): the minimum score necessary to be a viable solution, default 0.3
    * nms_threshold (float): the minimum nms value to be a viable solution, default: 0.4
    * compare_function (function): function that accepts two boxes and returns their overlap ratio, this function must
      accept two boxes and return an overlap ratio
    * eta (float): a coefficient in adaptive threshold formula: \ |nmsi1|\ =eta\*\ |nmsi0|\ , default: 1.0
    * top_k (int): if >0, keep at most top_k picked indices. default:0

    .. |nmsi0| replace:: nms_threshold\ :sub:`i`\

    .. |nmsi1| replace:: nms_threshold\ :sub:`(i+1)`\


    """

    if 'eta' in kwargs:
        eta = kwargs['eta']
    else:
        eta = 1.0
    assert 0 < eta <= 1.0

    if 'top_k' in kwargs:
        top_k = kwargs['top_k']
    else:
        top_k = 0
    assert 0 <= top_k

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = 0.3
    assert score_threshold > 0

    if 'nms_threshold' in kwargs:
        nms_threshold = kwargs['nms_threshold']
    else:
        nms_threshold = 0.4
    assert 0 < nms_threshold < 1

    if 'compare_function' in kwargs:
        compare_function = kwargs['compare_function']
    else:
        compare_function = None
        
        
    if 'overlap' in kwargs and (kwargs['overlap'] is not None):
        over = kwargs['overlap']
    else:
        over = None
     
        
    assert compare_function is not None

    if len(boxes) == 0:
        return []

    assert len(scores) == len(boxes)
    assert scores is not None

    # sort scores descending and convert to [[score], [indexx], . . . ]
    scores = get_max_score_index(scores, score_threshold, top_k)

    # Do Non Maximal Suppression
    # This is an interpretation of NMS from the OpenCV source in nms.cpp and nms.
    adaptive_threshold = nms_threshold
    indicies = []

    for i in range(0, len(scores)):
        idx = int(scores[i][1])
        keep = True
        for k in range(0, len(indicies)):
            if not keep:
                break
            kept_idx = indicies[k]

            iou = compare_function(boxes[idx], boxes[kept_idx], over)
            keep = (iou <= adaptive_threshold)
#             print(keep)

        if keep:
            indicies.append(idx)

        if keep and (eta < 1) and (adaptive_threshold > 0.5):
                adaptive_threshold = adaptive_threshold * eta

    return indicies
def nonMaximumSuppresion(objectProbas, contours, score_threshold=0.8, iou_threshold=0.7, overlap=None):
    """
    This function performs non maximum suppression on a single contour
    :param objectProbas: the object probabilities
    :param contours: the contours
    :param score_threshold: the score threshold to use
    :param iou_threshold: the IoU threshold to use
    :return: the non maximum suppressed contour
    """

    ids = nms(contours,
               objectProbas,
               score_threshold=score_threshold,
               nms_threshold=iou_threshold,
               compare_function=IOU2,
               overlap=overlap)

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
                 size=256,
                 kernel_size=1,
                 weight_decay=0,
                 weight_pos=0.2,
                 contourSize=291,
                 device="cuda:2",
                 n_channels=3,
                 patience=15,
                 scheduler="ReduceOnPlateauLR"
                 ):
        """
        """
        # initialize the parent class
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 3, 256, 256)
        self.MAX_CONTOUR_SIZE = contourSize
        global MAX_CONTOUR_SIZE
        MAX_CONTOUR_SIZE = contourSize
        # Modules of the model
        
        self.rgb2gray = nn.Conv2d(3,
                                  1,
                                  kernel_size=1,
                                  padding=0)
        
        self.segmentationBackbone = UNet(n_channels=n_channels, num_features=num_features)
        
        self.objectFinder = nn.Conv2d(num_features,
                                      num_classes,
                                      kernel_size=kernel_size,
                                      padding=kernel_size//2)

        self.objectOverlap = nn.Conv2d(num_features,
                                        1,
                                        kernel_size=kernel_size,
                                        padding=kernel_size//2)

        self.ControlPointsAngleRegressor = nn.Conv2d(num_features,
                                                     num_control_points,
                                                     kernel_size=kernel_size,
                                                     padding=kernel_size//2)

        self.ControlPointsDistanceRegressor = nn.Conv2d(num_features,
                                                        num_control_points,
                                                        kernel_size=kernel_size,
                                                        padding=kernel_size//2)

        x = torch.arange(size)
        y = torch.arange(size)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.shifts = torch.stack((xx, yy), dim=0)
        self.shifts2 = self.shifts.permute(1, 2, 0).reshape(1, 256, 256, 1, 2)
        self.cos = Cos()
        self.sin = Sin()
        
        self.step = 0
        self.warmup = 0
                
        weights = torch.FloatTensor([self.hparams["weight_pos"]])
        self.bce = nn.BCEWithLogitsLoss(weight=weights)

        self.B3M = getBsplineMatrix(numSamples=contourSize,
                                    degree=splineBasis,
                                    numControlPoints=num_control_points+3).float()
    def to(self, *args, **kwargs):
        print(args, kwargs)
        self = super().to(*args,**kwargs)
        self.rgb2gray = self.rgb2gray.to(*args,**kwargs)
        self.segmentationBackbone = self.segmentationBackbone.to(*args,**kwargs)
        self.objectFinder = self.objectFinder.to(*args,**kwargs)
        self.objectOverlap = self.objectOverlap.to(*args,**kwargs)
        self.ControlPointsAngleRegressor = self.ControlPointsAngleRegressor.to(*args,**kwargs)
        self.ControlPointsDistanceRegressor = self.ControlPointsDistanceRegressor.to(*args,**kwargs)
        self.shifts = self.shifts.to(*args,**kwargs)
        self.shifts2 = self.shifts2.to(*args,**kwargs)
        self.B3M = self.B3M.to(*args,**kwargs)
        return self
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
        parser.add_argument("--size", type=int, default=256)
        parser.add_argument("--kernel_size", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--weight_pos", type=float, default=0.2)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--scheduler", type=str, default="ReduceOnPlateau")


        return parent_parser

    def forward(self, x):
        """
        """
        
#         if self.hparams["n_channels"] == 3:
#             x = F.tanh(self.rgb2gray(x))
            
        features = self.segmentationBackbone(x)
        objectProbas = self.objectFinder(features)
        objectOverlap = self.objectOverlap(features)
        angles = self.ControlPointsAngleRegressor(features)
        distances = torch.abs(self.ControlPointsDistanceRegressor(features))
        controlPoints = torch.stack((self.cos(angles)*distances,
                                     self.sin(angles)*distances), dim=2) + self.shifts
        
        controlPoints = controlPoints.permute(0, 3, 4, 1, 2)
        controlPoints = torch.cat((controlPoints, controlPoints[:, :, :, :3, :]), 3)

        return objectProbas, angles, distances, controlPoints, objectOverlap


#     def matchInstances(self, selectedContours, selectedScores, targetContours):
#         """
#         """
#         # for increasing thresholds, compute the IoU pairwise between selected and target
#         cost = np.zeros((selectedContours.shape[0], targetContours.shape[0]))
#         for i in range(selectedContours.shape[0]):
#             for j in range(targetContours.shape[0]):
#                 cost[i, j] = polygon_iou(selectedContours[i], targetContours[j])
        
#         # solve linear assignement problem
#         rows, cols = linear_sum_assignment(-cost)
# #         values = -values

#         values = cost[rows, cols]
    
#         precisions, recalls = [], []
#         thresholds = np.linspace(0, 1, 10)
        
#         for threshold in thresholds:
#             # compute true positifs, false negatives, false negatives
#             FN = (values == 0).sum()
#             TP = (values>=threshold).sum()- FN
#             FP = (values<threshold).sum() - FN
            
            
#             # compute precision, recall, f1
#             precision = TP/(TP+FP)
#             recall = TP/(TP+FN)

#             precisions.append(precision)
#             recalls.append(recall)

#         # # find the best threshold
#         # bestThreshold = thresholds[np.argmax(precisions)]
#         # # compute the best precision and recall
#         # bestPrecision = precisions[np.argmax(precisions)]
#         # bestRecall = recalls[np.argmax(precisions)]


#         return precisions, recalls, thresholds

    def matchInstances(self, selectedContours, selectedScores, targetContours):
        """
        """
        precisions, recalls = np.zeros((10, 10)), np.zeros((10, 10))
        IoU_thresholds = np.linspace(0.2, 0.8, 10)
        confidence_thresholds = np.linspace(0, 1, 10)[::-1]

        # for increasing thresholds, compute the IoU pairwise between selected and target
        cost = np.zeros((selectedContours.shape[0], targetContours.shape[0]))
        for i in range(selectedContours.shape[0]):
            for j in range(targetContours.shape[0]):
                cost[i, j] = polygon_iou(selectedContours[i], targetContours[j])
        
        for j, confidence_threshold in enumerate(confidence_thresholds):
        

            cost_filtered = cost[selectedScores > confidence_threshold, :]

            rows, cols = linear_sum_assignment(-cost_filtered)
            values = cost_filtered[rows, cols]
    

            for i, threshold in enumerate(IoU_thresholds):
                # compute true positifs, false negatives, false negatives
                FN = float((values == 0).sum())
                TP = float((values>=threshold).sum())
                FP = float((values<threshold).sum() - FN)
                
                
                # compute precision, recall, f1
#                 if(TP == 0):
#                     precision = 0
#                     recall = 1
#                 else:
                try:
                    precision = TP/(TP+FP)
                except:
                    precision = 1

                recall = TP/(targetContours.shape[0])

                    
#                 if(precision == 0 or recall == 0):
#                     f1 = 0
#                 else:
#                     f1 = 2*precision*recall/(precision+recall)

                # precisions.append(precision)
                # recalls.append(recall)
                precisions[i, j] = precision
                recalls[i, j] = recall

        return precisions, recalls, IoU_thresholds


    def compute_loss(self, predicted_targets, targets, warmup=0):
        """
        This function computes the loss of the model
        :param predicted_targets: the predicted targets
        :param targets: the targets
        """
        (objectProbas, controlPoints, contours, objectOverlapProbas) = predicted_targets
        (targetObjectProbas, targetOverlapProbas, targetContours) = targets

        # loss 1

        #lossObjectProba = F.binary_cross_entropy(objectProbas, targetObjectProbas, pos_weight=weights)
        mask = targetOverlapProbas < 0.5
        lossObjectProba = self.bce(objectProbas[mask], targetObjectProbas[mask])
        lossObjectOverlapProba = self.bce(objectOverlapProbas, targetOverlapProbas)
#         lossObjectProba = F.mse_loss(objectProbas, targetObjectProbas)
        lossContour = computeContourLoss(objectProbas,
                                         contours,
                                         targetObjectProbas,
                                         targetOverlapProbas,
                                         targetContours,
                                         lambda2=self.hparams["lambda2"],
                                         shifts=self.shifts2)
        
        finalLoss = lossObjectProba + self.hparams["lambda1"]*lossContour + lossObjectOverlapProba

        return finalLoss

        
        
    def training_step(self, batch, batch_idx):
        """
        This function is called for each batch
        :param batch: the batch
        :param batch_idx: the batch index
        """

        images, targets = batch
        objectProbas, angles, distances, controlPoints, objectOverlap = self(images)
        
        contours = getContourSamples(controlPoints, self.B3M)
        contours = contours.permute(1, 2, 3, 0, 4)#.flatten(1, 2)#reshape(objectProbas.shape[0], -1, MAX_CONTOUR_SIZE, 2)
        
        predicted_targets = (objectProbas, controlPoints, contours, objectOverlap)

#         if self.trainer.global_step > 500:
#             self.warmup = 1

        loss = self.compute_loss(predicted_targets, targets, 1)
        self.log("loss_train", loss)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        
        tsb = self.logger.experiment
        
        images, targets = batch
        objectProbas, angles, distances, controlPoints, objectOverlap = self(images)
        
        contours = getContourSamples(controlPoints, self.B3M)
        contours = contours.permute(1, 2, 3, 0, 4)#.flatten(1, 2)#reshape(objectProbas.shape[0], -1, MAX_CONTOUR_SIZE, 2)
        
        predicted_targets = (objectProbas, controlPoints, contours, objectOverlap)

        loss = self.compute_loss(predicted_targets, targets)
        self.log("loss_val", loss)
        self.log("hp_metric", loss)


        if(batch_idx == 0):
            ####################################################
            
            tsb.add_histogram("Weights/objectFinder", self.objectFinder.weight, self.step)
            tsb.add_histogram("Weights/AnglesRegressor", self.ControlPointsAngleRegressor.weight, self.step)
            tsb.add_histogram("Weights/DistanceRegressor", self.ControlPointsDistanceRegressor.weight, self.step)
            
            
            
            tsb.add_histogram('distances', distances, self.step)
            tsb.add_histogram('angles', angles, self.step)
            tsb.add_histogram('objectProbas', objectProbas, self.step)
            ####################################################
            fig = plt.figure(figsize=(10, 10))
            objectProbas = torch.sigmoid(objectProbas)
            im = objectProbas[0, 0].cpu().numpy()
            plt.imshow(im, cmap="gnuplot2")
            plt.colorbar()
            tsb.add_figure("Sample/objectProbas", fig, self.step)      
            

            fig = plt.figure(figsize=(10, 10))
            objectOverlapProbas = torch.sigmoid(objectOverlap)
            im = objectOverlapProbas[0, 0].cpu().numpy()
            plt.imshow(im, cmap="gnuplot2")
            plt.colorbar()
            tsb.add_figure("Sample/objectOverlapProbas", fig, self.step)    
            ###################################################
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            ax.imshow(np.zeros((256, 256)), cmap="gray")
            ct = controlPoints[0].cpu().numpy()
            for i in range(0, 256, 8):
                for j in range(0, 256, 8):
                    ax.fill(ct[i, j, :, 0], ct[i, j, :, 1], 'o-')
            # plot_to_tensorboard(tsb, fig, None, f"densePredictionsConvexHull")
            tsb.add_figure(f"Sample/densePredictionsConvexHull", fig, self.step)
            ###################################################
            
            
            for i in range(self.hparams["num_control_points"]):
                circular_angle = angles[0, i].detach().cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
                ax1.imshow(distances[0, i].detach().cpu().numpy())
                norm = mpl.colors.Normalize(0.0, 2*np.pi)
                colormap = plt.get_cmap('hsv')

                ax2.imshow(circular_angle, norm=norm, cmap=colormap)
                # plot_to_tensorboard(tsb, fig, None, f"distance_angles{i}")
                tsb.add_figure(f"distance_angles{i}", fig, self.step)
            
            contours = contours.flatten(1, 2)
            threshold = self.hparams["object_threshold"]
            nms_threshold = self.hparams["nms_threshold"]
            contours = contours[0].reshape(-1, self.MAX_CONTOUR_SIZE, 2)

            xmin = torch.amin(contours[:, :, 1], dim=-1) 
            xmax = torch.amax(contours[:, :, 1], dim=-1)
            ymin = torch.amin(contours[:, :, 0], dim=-1)
            ymax = torch.amax(contours[:, :, 0], dim=-1)

            bboxes = torch.stack([xmin, ymin, xmax, ymax], -1)

            scores = objectProbas[0].reshape(256*256)

            bboxes = bboxes[scores>threshold]
            scores2 = scores[scores>threshold]

            selectedIds = batched_nms(bboxes, scores2, torch.ones(len(scores2)), iou_threshold=self.hparams["nms_threshold"])

            bboxes = bboxes.detach().cpu().numpy()
            selectedIds = selectedIds.detach().cpu().numpy()

            rects = bboxes[selectedIds]
            
            
#             scores = objectProbas[0].flatten(1).cpu().detach().numpy()
#             sortedInstances = np.argsort(scores, axis=1).copy()

#             til = [np.searchsorted(scores[0, sortedInstances[0]], threshold)]

#             contours2 = [contours[til[0]:].detach().cpu().numpy()]

#             print(len(contours2[0]))
#             postProcessed = nonMaximumSuppresion(scores[0, sortedInstances[0, til[0]:]],
#                                                 contours2[0],
#                                                 threshold,
#                                                 nms_threshold)

#             contoursSelected, selectedIds = postProcessed
#             selectedIds = np.array(selectedIds)


#             image2 = denormalize(images[0].detach().cpu()).numpy().transpose(1, 2, 0)
    
            image2 = np.uint8(denormalize(images[0].detach().cpu()).numpy().transpose(1, 2, 0)*255)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #     image2 = clahe.apply(image2)
            img_yuv = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            image2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    
            ctrs = contours[scores>threshold][selectedIds].detach().cpu().numpy()
#             ctrs = contoursSelected


            fig = plt.figure(figsize=(10, 10))
            ax4 = fig.gca()
            ax4.imshow(image2)

            for j, ct in enumerate(ctrs):
                ax4.fill(ctrs[j][:, 0], ctrs[j][:, 1], color=colors[j], alpha=0.3)
                ax4.plot(ctrs[j][:, 0], ctrs[j][:, 1],color=colors[j], linewidth=2, linestyle='dashed')


            tsb.add_figure(f"Sample/Instances", fig, self.step)
                
            

            ########################### Draw precision recall curve ###############################

            (targetObjectProbas, targetOverlapProba, targetObjectContours) = targets
            

            targetObjectContours = targetObjectContours[0].reshape(256*256, -1, 2)
            targetObjectContours = torch.unique(targetObjectContours, dim=0)[1:]
            


            precisions, recalls, thresholds = self.matchInstances(contours[scores>threshold][selectedIds],
                                                                   scores[scores>threshold][selectedIds].cpu().numpy(),
                                                                   targetObjectContours)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            for i in range(precisions.shape[0]):
                ax.plot(recalls[i], precisions[i], "o-", label=f"IoU@{thresholds[i]:.2f}")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend()
#             ax.set_ylim([0.0, 1.05])
#             ax.set_xlim([0.0, 1.0])
            ax.grid(True)

            tsb.add_figure(f"Sample/PrecisionRecall", fig, self.step)
            

            self.step += 1
                
        return {"loss": loss}


    def configure_optimizers(self):
        """

        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
#                                                                         T_0=10,
#                                                                         T_mult=2)
        if self.hparams["scheduler"] == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)
        lr_schedulers = {"scheduler": scheduler, "monitor": "loss_val"}
        return [optimizer], lr_schedulers
