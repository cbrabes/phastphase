# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:05:39 2025

@author: sivan
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def Fourier(x):
    N = np.shape(x)[0]*np.shape(x)[1]
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x))) / np.sqrt(N)

def integer(n):
    '''return the rounded integer (if you cast a number as int, it will floor the number)'''
    return int(np.round(n))

def set_center(image, center):
    '''
    this centering routine shifts the image in a cyclical fashion
    INPUT:  image: array, difference hologram
            center: array, center coordinates [x, y]
    OUTPUT: centered hologram
    -------
    '''
    xdim, ydim = image.shape
    xshift = integer(xdim / 2 - center[1])
    yshift = integer(ydim / 2 - center[0])
    image_shift = np.roll(image, yshift, axis=0)
    image_shift = np.roll(image_shift, xshift, axis=1)
    #print('Shifted image by %i pixels in x and %i pixels in y.'%(xshift, yshift))
    return image_shift

def plot_all(reconstruction, s = 0):
    image = np.array(reconstruction.copy())
    if s>0:
        [i,j] = np.unravel_index(np.argmax(np.abs(image)),np.shape(image))
        image[i-s:i+s,j-s:j+s] = 0
    _, ax = plt.subplots(1,2, figsize = (12,4))
    im0 = ax[0].imshow(np.abs(image), cmap="jet")
    #plt.colorbar(im0, ax=ax[1], fraction=0.046, pad=0.04)
    ax[0].set_title('Reconstructed amplitude')
    im1 = ax[1].imshow(np.angle(image),cmap="twilight", interpolation="none")
    ax[1].set_yticks([])
    #plt.colorbar(im1, ax=ax[2], fraction=0.046, pad=0.04)
    ax[1].set_title('Reconstructed phase')
    plt.show()

def sort_contour_points(points):
    """
    Sorts the given points in counterclockwise order around the centroid of the polygon.

    Parameters:
        points (ndarray): Nx2 array of (x, y) coordinates.

    Returns:
        ndarray: Sorted points in counterclockwise order.
    """
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on the calculated angles
    sorted_indices = np.argsort(angles)

    return points[sorted_indices]

def find_corners(coords, threshold = np.pi / 4, threshold2 = np.pi , PLOT = False):

    # Compute differences (dx, dy) between consecutive points
    dx = np.diff(coords[:, 1], append=coords[0, 0])
    dy = np.diff(coords[:, 0], append=coords[0, 1])

    # Compute angle changes between consecutive segments
    angles = np.arctan2(dy, dx)
    angle_changes = np.diff(angles, append=angles[0])

    # Find indices where the angle change exceeds a threshold (sharp corners)
    angle_changes[np.abs(angle_changes) >threshold2] = 0
    corner_indices = np.where(np.abs(angle_changes) > threshold)[0]

    if PLOT:
        plt.figure()
        plt.plot(angle_changes)
        plt.plot([0,len(angle_changes)],[-threshold,-threshold])

    return coords[corner_indices]

def bound_image(image):
    # Label connected regions
    labeled_image = label(image)
    
    # Find bounding box coordinates for all regions together
    regions = regionprops(labeled_image)
    min_row = min(region.bbox[0] for region in regions)
    min_col = min(region.bbox[1] for region in regions)
    max_row = max(region.bbox[2] for region in regions)
    max_col = max(region.bbox[3] for region in regions)
    
    support = image[min_row:max_row,min_col:max_col]
    return support