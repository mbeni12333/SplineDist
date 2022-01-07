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
from torchvision.ops import box_iou, batched_nms
import matplotlib.patches as patches
from torch import cdist
from .unet import UNet
from utils import *
import tqdm
import pdb
from torchvision.utils import make_grid


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



def computeContourLoss(objectProbas, contours, targetObjectProbas, targetContours, nms_threshold=0.7, obj_threshold=0.7, lambda2=1.0, shifts=None):
    """
    """
    targetObjectProbas = targetObjectProbas.squeeze(1)
    mask = (targetObjectProbas > 0)
    mask_norm = mask.flatten(1).sum(1).reshape(-1, 1, 1)
    N = targetObjectProbas.shape[0]
    


    
    term1 = (targetObjectProbas * mask * torch.abs(contours - targetContours).mean((-1, -2))/mask_norm).sum()/N
    term2 = ((~mask) * torch.abs(contours - shifts.permute(1, 2, 0).reshape(1, 256, 256, 1, 2)).mean((-1, -2))/(256*256-mask_norm)).sum()/N
    
    #print(f"mask {mask.shape},\
    #mask_norm {mask_norm},\
    #term1 {term1},\
    #term2 {term2}")

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
                 size=256,
                 kernel_size=1,
                 device="cuda:2"):
        """
        """
        # initialize the parent class
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 3, 256, 256)

        # Modules of the model
        self.segmentationBackbone = UNet(num_features=num_features)
        
        self.objectFinder = nn.Conv2d(num_features,
                                      num_classes,
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
        xx, yy = torch.meshgrid(x, y)
        self.shifts = torch.stack((xx, yy), dim=0).to(device)
        self.cos = Cos()
        self.sin = Sin()
        
        self.step = 0
                
        weights = torch.FloatTensor([0.2])
        self.bce = nn.BCEWithLogitsLoss(weight=weights)

        self.B3M = getBsplineMatrix(numSamples=MAX_CONTOUR_SIZE,
                                    degree=splineBasis,
                                    numControlPoints=num_control_points+3).float().to(device)
    
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
        """
        """
        
        
        features = self.segmentationBackbone(x)
        objectProbas = self.objectFinder(features)
        angles = self.ControlPointsAngleRegressor(features)
        distances = self.ControlPointsDistanceRegressor(features)
        controlPoints = torch.stack((self.cos(angles)*distances,
                                     self.sin(angles)*distances), dim=2) + self.shifts
        
        controlPoints = controlPoints.permute(0, 3, 4, 1, 2)
        controlPoints = torch.cat((controlPoints, controlPoints[:, :, :, :3, :]), 3)

        return objectProbas, angles, distances, controlPoints


    def matchInstances(self, selectedContours, selectedScores, targetContours):
        """
        """
        # for increasing thresholds, compute the IoU pairwise between selected and target
        cost = np.zeros(selectedContours.shape[0], targetContours.shape[0])
        for i in range(selectedContours.shape[0]):
            for j in range(targetContours.shape[0]):
                cost[i, j] = polygon_iou(selectedContours[i], targetContours[j])
        
        # solve linear assignement problem
        values, selectedIds = linear_assignment(cost)


        precisions, recalls = [], []
        thresholds = np.linspace(0, 1, 20)

        for threshold in thresholds:
            # compute true positifs, false negatives, false negatives
            TP = (values>=threshold).sum()
            FN = len(set(range(targetContours.shape[0])) - TP)
            FP = (values<threshold).sum()
            # compute precision, recall, f1
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)

            precisions.append(precision)
            recalls.append(recall)

        # # find the best threshold
        # bestThreshold = thresholds[np.argmax(precisions)]
        # # compute the best precision and recall
        # bestPrecision = precisions[np.argmax(precisions)]
        # bestRecall = recalls[np.argmax(precisions)]


        return zip(precisions, recalls), thresholds


    def compute_loss(self, predicted_targets, targets):
        """
        This function computes the loss of the model
        :param predicted_targets: the predicted targets
        :param targets: the targets
        """
        (objectProbas, controlPoints, contours) = predicted_targets
        (targetObjectProbas, targetOverlapProbas, targetContours) = targets

        # loss 1

        #lossObjectProba = F.binary_cross_entropy(objectProbas, targetObjectProbas, pos_weight=weights)
        lossObjectProba = self.bce(objectProbas, targetObjectProbas)
#         lossObjectProba = F.mse_loss(objectProbas, targetObjectProbas)
        lossContour = computeContourLoss(objectProbas,
                                         contours,
                                         targetObjectProbas,
                                         targetContours,
                                         lambda2=self.hparams["lambda2"],
                                         shifts=self.shifts)
        
        finalLoss = lossObjectProba + self.hparams["lambda1"]*lossContour

        return finalLoss

        
        
    def training_step(self, batch, batch_idx):
        """
        This function is called for each batch
        :param batch: the batch
        :param batch_idx: the batch index
        """

        images, targets = batch
        objectProbas, angles, distances, controlPoints = self(images)
        
        contours = getContourSamples(controlPoints, self.B3M)
        contours = contours.permute(1, 2, 3, 0, 4)#.flatten(1, 2)#reshape(objectProbas.shape[0], -1, MAX_CONTOUR_SIZE, 2)
        
        predicted_targets = (objectProbas, controlPoints, contours)
        loss = self.compute_loss(predicted_targets, targets)
        self.log("train_loss", loss)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        
        tsb = self.logger.experiment
        
        images, targets = batch
        objectProbas, angles, distances, controlPoints = self(images)
        
        contours = getContourSamples(controlPoints, self.B3M)
        contours = contours.permute(1, 2, 3, 0, 4)#.flatten(1, 2)#reshape(objectProbas.shape[0], -1, MAX_CONTOUR_SIZE, 2)
        
        predicted_targets = (objectProbas, controlPoints, contours)

        loss = self.compute_loss(predicted_targets, targets)
        self.log("val_loss", loss)
        self.log("hp_metric", loss)


        if(batch_idx == 0):
            ####################################################
            
            tsb.add_histogram("Weights/objectFinder", self.objectFinder.weight, self.step)
            tsb.add_histogram("Weights/AnglesRegressor", self.ControlPointsAngleRegressor.weight, self.step)
            tsb.add_histogram("Weights/DistanceRegressor", self.ControlPointsDistanceRegressor.weight, self.step)
            
            
            
            tsb.add_histogram('Values/distances', distances, self.step)
            tsb.add_histogram('Values/angles', angles, self.step)
            tsb.add_histogram('Values/objectProbas', objectProbas, self.step)
            ####################################################
            fig = plt.figure(figsize=(10, 10))
            objectProbas = torch.sigmoid(objectProbas)
            im = objectProbas[0, 0].cpu().numpy()
            plt.imshow(im, cmap="gnuplot2")
            plt.colorbar()
            tsb.add_figure("Sample/objectProbas", fig, self.step)      
            
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
                circular_angle = np.mod(angles[0, i].detach().cpu().numpy() + np.pi, 2*np.pi)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                ax1.imshow(distances[0, i].detach().cpu().numpy())
                ax2.imshow(circular_angle, cmap="gist_rainbow")
                # plot_to_tensorboard(tsb, fig, None, f"distance_angles{i}")
                tsb.add_figure(f"Sample/distance_angles{i}", fig, self.step)
            
            contours = contours.flatten(1, 2)
            threshold = self.hparams["object_threshold"]
            nms_threshold = self.hparams["nms_threshold"]
            contours = contours[0].reshape(-1, MAX_CONTOUR_SIZE, 2)

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


            #image2 = denormalize(images[0].detach().cpu()).numpy().transpose(1, 2, 0)
    
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
                
            
            self.step += 1
                
        return {"loss": loss}


    def configure_optimizers(self):
        """

        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=0.0001)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 40)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
#         lr_schedulers = {"scheduler": scheduler, "monitor": "val_loss"}
        return [optimizer], scheduler#, lr_schedulers
