import torch
from scipy.interpolate import splev
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from skimage import exposure


def denormalize(x):
    """
    Denormalize the image
    :param x: the image to denormalize
    :return: the denormalized image
    """
    mean, std = 0.5, 0.5
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    x = transforms.Normalize(mean=mean_inv, std=std_inv)(x)
    x = x.clamp(0, 1)
    return x


def getRandomColor():
    """
    Generate a random color
    :return: an array of 3 random values between 0 and 255
    """
    color = np.random.randint(30, 255, size=3)
    color = (int(color[0]), int(color[1]), int(color[2]))
    return color


def getInstancesImageFromContours(contours, shape=(256, 256), color=None):
    """
    Get the instances image from the contours
    :param contours: the contours
    :return: the instances image
    """
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for contour in contours:
        color = getRandomColor()
        # print(contour.shape, contour.dtype, contour.min(), contour.max())
        cv2.fillPoly(image, pts=[contour], color=color)
    return image


def sampleBsplineFromControlPoints(controlPoints, numSamples, degree=3):
    """
    Sample a bspline from the control points
    :param controlPoints: the control points
    :param numSamples: the number of samples
    :param degree: the degree of the bspline
    :return: the sampled bspline
    """

    k = degree
    t = np.zeros((len(controlPoints)+degree*2+1))
    t[:degree] = 0
    t[degree:-degree] = np.linspace(0, 1, len(t) - degree*2)
    t[-degree:] = 1

    def B(x, k, i, t):
        """
        The B-spline function
        :param x: the x value
        :param k: the degree of the bspline
        :param i: the index of the control point
        :param t: the knot vector
        :return: the value of the bspline
        """
        if k == 0:
            return 1.0 if t[i] <= x < t[i+1] else 0.0

        if t[i+k] == t[i]:
            c1 = 0.0

        else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)

        if t[i+k+1] == t[i+1]:
            c2 = 0.0

        else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
        return c1 + c2

    def bspline(x, t, c, k):
        """
        Evaluate the bspline at x
        :param x: the x value
        :param t: the knot vector
        :param c: the control points
        :param k: the degree of the bspline
        :return: the value of the bspline
        """
        n = len(c)
        assert (n >= k+1) and (len(c) >= n)
        b = torch.FloatTensor([B(x, k, i, t) for i in range(n)]).reshape(1, n)
        return b @ c

    u = np.linspace(0, 1, numSamples, endpoint=True)
    controlPoints = torch.vstack((controlPoints, controlPoints[:3]))
    return torch.vstack([bspline(x, t, controlPoints, k) for x in u][degree:-degree])


def showBatch(batch):
    """
    Show a batch of images
    :param batch: the batch of images
    :return:
    """
    batch_x, batch_y = batch
    # for i in range(batch_x.shape[0]):
    #     plt.figure()
    #     plt.imshow(batch_x[i,0,:,:])
    #     plt.imshow(batch_y[i,0,:,:], alpha=0.5)
    #     plt.show()
    for item in range(len(batch_x)):

        img = batch_x[item]

        (objectProbas, overlapProba, objectContours,
            mask) = batch_y[item].values()

        # plt.figure(figsize=(20, 10))
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            1, 5, figsize=(25, 5))

        # plt.subplot(1,4,1)
        ax1.set_title("Original Image")
        ax1.imshow(denormalize(img).permute(1, 2, 0).clamp(0, 1))

        ax2.set_title("Image With Adaptive histogram Equalization")
        eq = exposure.equalize_adapthist((denormalize(img).permute(
            1, 2, 0).clamp(0, 1).numpy()*256).astype(np.uint8), clip_limit=0.03)
        ax2.imshow(np.clip(eq, 0, 1))
        # plt.subplot(1,4,2)
        ax3.set_title("Object Probabilities")
        print(objectProbas.min(), objectProbas.max(), objectProbas.ptp())
        ax3.imshow(objectProbas.clip(0, 1), cmap='gnuplot2')
        # print(objectProbas.max())
        #ax2.hist(objectProbas[objectProbas > 0].flatten(), bins=100)
        # plt.subplot(1,4,3)
        ax5.set_title("Overlap Probabilities")
        ax5.imshow(overlapProba.clip(0, 1), cmap='gnuplot2')
        # show overlap histogram
        #ax3.hist(overlapProba.flatten(), bins=100)

        # plt.subplot(1,4,4)
        ax4.set_title("Object Instances")
        instances = getInstancesImageFromContours(objectContours)
        ax4.imshow(instances)

        fig.tight_layout()
        plt.show()
