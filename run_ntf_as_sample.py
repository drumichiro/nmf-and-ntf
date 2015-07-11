#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import pylab as plt
import matplotlib as mlib
from mpl_toolkits.mplot3d import Axes3D
import ntf as ntflib
from myutil.sampler import *

###########################################

###########################################


def swapIndexForMeshGrid3d(array):
    temp = np.copy(array[1])
    array[1] = array[0]
    array[0] = temp
    # array[0], array[1] = np.copy(array[1]), np.copy(array[0])
    return array


def createIndexFromEdge(edge):
    mid = np.array([i1[1:] + i1[:-1] for i1 in edge])/2
    mid = swapIndexForMeshGrid3d(mid)  # This swap may not be necessary.
    grid = np.meshgrid(*mid)
    index = np.array([i1.ravel() for i1 in grid])
    index = swapIndexForMeshGrid3d(index)  # This swap may not be necessary.
    return index


def normalize(value):
    eps = 0.000000001
    value = value + eps
    valueMin = np.min(value)
    valueMax = np.max(value)
    return (value - valueMin)/(valueMax - valueMin + eps)


def transformHistToColorStrength(value):
    value = normalize(value).ravel()
    # This color map start from blue and terminate in red.
    offset = 0.6
    value = (1 - value)*offset

    # Added S and V of HSV.
    length = len(value)
    hsv = np.vstack([value, np.ones(length), np.ones(length)]).T
    rgb = mlib.colors.hsv_to_rgb(hsv)
    return rgb


def transformHistToSize(value):
    offset = 10
    return normalize(value).ravel()*(2000 - offset) + offset


def showDistribution(index, color, size):
    i, j, k = index
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(i, j, k, color=color, s=size, marker="o")
    plt.show()


def showHistDistribution(hist, edge):
    index = createIndexFromEdge(edge)
    color = transformHistToColorStrength(hist)
    size = transformHistToSize(hist)
    showDistribution(index, color, size)


def transformSampleToHist(x, mu, sigma):
    sigmaAve = []
    for i1 in sigma:
        sigmaAve.append(np.trace(i1)/len(i1))
    rangeMin = np.min(mu, axis=0) - 2*np.max(sigmaAve, axis=0)
    rangeMax = np.max(mu, axis=0) + 2*np.max(sigmaAve, axis=0)
    histRange = np.array([rangeMin, rangeMax]).T
    return np.histogramdd(x, bins=(5, 5, 5), range=histRange)


if __name__ == '__main__':
    # Generate samples as input data from Gaussians.
#     mu = [[10, 20, 30],
#           [20, 30, 10],
#           [30, 10, 20]]
#     sigma = [[[5, 0, 0],
#               [0, 5, 0],
#               [0, 0, 5]],
#              [[5, 0, 0],
#               [0, 5, 0],
#               [0, 0, 5]],
#              [[5, 0, 0],
#               [0, 5, 0],
#               [0, 0, 5]]]
    mu = [[20, 20, 20]]
    sigma = [[[5, 0, 0],
              [0, 5, 0],
              [0, 0, 5]]]
    sampleNum = 100
    x = generateVectorSample(sampleNum, mu, sigma)
    hist, edge = transformSampleToHist(x, mu, sigma)
    # showHistDistribution(hist, edge)

    # Use \# of Gaussians as \# of classes.
    classNum = len(mu)
    # Start factorization.
    ntf = ntflib.NTF(classNum, hist)
    ntf.factorize(hist)
    hist = ntf.reconstruct()
    factor = ntf.getFactor()

    showHistDistribution(hist, edge)
