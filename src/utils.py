import torch
# from scipy.interpolate import splev
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import colorsys
# from skimage import exposure
from matplotlib.patches import Polygon


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
    h, s, l = np.random.random(
        size=3)/np.array([1.0, 2.0, 5.0]) + np.array([0.0, 0.5, 0.4])
    # random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    # r, g, b = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
    # colorsys.hsv_to_rgb()
    # color = (int(color[0]), int(color[1]), int(color[2]))
    # return color
    return (r, g, b)


def getInstancesImageFromContours(contours, shape=(256, 256), color=None):
    """
    Get the instances image from the contours
    :param contours: the contours
    :return: the instances image
    """
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    colors = [getRandomColor() for i in range(len(contours))]
    for i, contour in enumerate(contours):
        cv2.fillPoly(image, pts=[contour], color=colors[i])
    return image, colors


def B(x, k, i, t):

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

# def sampleBsplineFromControlPoints(controlPoints, numSamples, degree=3):
#     """
#     Sample a bspline from the control points
#     :param controlPoints: the control points
#     :param numSamples: the number of samples
#     :param degree: the degree of the bspline
#     :return: the sampled bspline
#     """

#     k = degree
#     t = np.zeros((len(controlPoints)+degree*2+1))
#     t[:degree] = 0
#     t[degree:-degree] = np.linspace(0, 1, len(t) - degree*2)
#     t[-degree:] = 1

#     def bspline(x, t, c, k):
#         """
#         Evaluate the bspline at x
#         :param x: the x value
#         :param t: the knot vector
#         :param c: the control points
#         :param k: the degree of the bspline
#         :return: the value of the bspline
#         """
#         n = len(c)
#         assert (n >= k+1) and (len(c) >= n)
#         b = torch.FloatTensor([B(x, k, i, t) for i in range(n)]).reshape(1, n)
#         return b @ c

#     u = np.linspace(0, 1, numSamples, endpoint=True)
#     controlPoints = torch.vstack((controlPoints, controlPoints[:3]))
#     return torch.vstack([bspline(x, t, controlPoints, k) for x in u][degree:-degree])



def getBsplineMatrix(numSamples=1000, degree=3, numControlPoints=18):
    """
    """
    m = numControlPoints+degree+1
    knots = np.zeros(m+1)
    knots[1:] = np.arange(1, m+1)/m
    parametrization = np.linspace(knots[degree], knots[numControlPoints], numSamples, endpoint=True)
    # tt, kk = np.meshgrid(T, K)
    B3M = np.zeros((numSamples, numControlPoints))
    for i in range(numControlPoints):
        for j in range(numSamples):
            B3M[j, i] = B(parametrization[j], degree, i, knots)
    return torch.tensor(B3M)
    
def getContourSamples(controlPoints, B3M):
    """
    """
    return torch.tensordot(B3M, controlPoints, dims=([1], [3]))



def showBatch(batch):
    """
    Show a batch of images
    :param batch: the batch of images
    :return:
    """
    batch_x, batch_y = batch
    
    
    for item in range(len(batch_x)):

        img = batch_x[item]
        img = denormalize(img).permute(1, 2, 0).clamp(0, 1).numpy()
        print(img.ptp())
        (objectProbas, overlapProba, objectContours) = batch_y
        
        objectProbas = objectProbas[item].permute(1, 2, 0)
        overlapProba = overlapProba[item].permute(1, 2, 0)
        objectContours = objectContours[item]

        fig, (ax1, ax3, ax4, ax5) = plt.subplots(
            1, 4, figsize=(25, 8))

        # plt.subplot(1,4,1)
        ax1.set_title("Original Image")
        ax1.imshow(img)

#         ax2.set_title("Image With Adaptive histogram Equalization")
# #         eq = exposure.equalize_adapthist(img, clip_limit=0.03)
# #         ax2.imshow(np.clip(eq, 0, 1))
#         ax2.imshow(img)
        # plt.subplot(1,4,2)
        ax3.set_title("Object Probabilities")
        # print(objectProbas.min(), objectProbas.max(), objectProbas.ptp())
        ax3.imshow(objectProbas, cmap='gnuplot2')
        # plt.subplot(1,4,4)
        ax4.set_title("Object Instances")
        # instances, colors = getInstancesImageFromContours(objectContours)
        ax4.imshow(img)

        # ax4.imshow(instances, alpha=0.3)
        colors = [ list(map(lambda x: x/255, getRandomColor())) for i in range(len(objectContours))]
        for(i, contour) in enumerate(objectContours):
            ax4.fill(contour[:,0], contour[:,1], color=colors[i], alpha=0.3)
            ax4.plot(contour[:, 0], contour[:, 1],
                     color=colors[i], linewidth=2, linestyle='dashed')

        ax5.set_title("Overlap Probabilities")
        ax5.imshow(overlapProba, cmap="gray")

        fig.tight_layout()
        plt.show()

        
def plot_to_tensorboard(writer, fig, step, fig_name):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(fig_name, img, step)
    plt.close(fig)