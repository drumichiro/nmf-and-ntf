#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import numpy as np
import pylab as plt
import matplotlib as mlib
from mpl_toolkits.mplot3d import Axes3D
import ntf
from myutil.sampler import generateVectorSample

###########################################
EPS = 0.000000001
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
    value = value + EPS
    valueMin = np.min(value)
    valueMax = np.max(value)
    return (value - valueMin)/(valueMax - valueMin + EPS)


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


def show2dDistribution(index, color, size):
    i, j = index
    plt.scatter(i, j, color=color, s=size, marker="o")
    plt.show()


def show3dDistribution(index, color, size):
    i, j, k = index
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(i, j, k, color=color, s=size, marker="o")
    plt.show()


def showDistribution(index, color, size):
    dimention = len(index)
    if 2 == dimention:
        show2dDistribution(index, color, size)
    elif 3 == dimention:
        show3dDistribution(index, color, size)
    else:
        print "%dD scatter plot is unsupported." % dimention


def showHistDistribution(hist, edge):
    index = createIndexFromEdge(edge)
    color = transformHistToColorStrength(hist)
    size = transformHistToSize(hist)
    showDistribution(index, color, size)


def getMaxFactor(factor):
    maxFactor = []
    for fct1 in factor:
        for fct2 in fct1:
            maxFactor.append(np.max(fct2))
    return np.max(maxFactor)


def showFactorValue(factor):
    fig = plt.figure(figsize=(16, 12))
    fig.text(0.5, 0.04, 'Order', ha='center')
    fig.text(0.04, 0.5, 'Bases',
             va='center', rotation='vertical')
    colorList = ['b', 'g', 'r']
    colorLists = len(colorList)
    xLen, yLen = factor.shape[:2]
    elements = map(len, factor[0])
    upperLimit = getMaxFactor(factor)
    index = 0
    for i1 in np.arange(xLen):
        color = colorList[i1 % colorLists]
        for i2 in np.arange(yLen):
            index += 1
            ax = fig.add_subplot(xLen, yLen, index)
            ax.set_ylim([0, upperLimit])
            line = np.arange(elements[i2])
            ax.bar(line, factor[i1][i2], color=color)
    plt.show()


def transformSampleToHist(x, mu, sigma, bins):
    sigmaAve = []
    for i1 in sigma:
        sigmaAve.append(np.trace(i1)/len(i1))
    rangeMin = np.min(mu, axis=0) - 2*np.max(sigmaAve, axis=0)
    rangeMax = np.max(mu, axis=0) + 2*np.max(sigmaAve, axis=0)
    histRange = np.array([rangeMin, rangeMax]).T
    return np.histogramdd(x, bins=bins, range=histRange)


def generateInitialFactorValue(mu, edge, classNum):
    initialFactor = []
    average = np.array(mu).T
    # Loop times which number
    for eachAve, eachEdge in zip(average, edge):
        tmp = []
        for i1 in np.arange(classNum):
            # Loop used average if classNum is larger than len(eachAve).
            i1 = i1 % len(eachAve)
            filledPos = np.digitize([eachAve[i1]], eachEdge) - 1
            # Fill one at average position.
            value = np.zeros(len(eachEdge) - 1)
            # value[filledPos-2:filledPos+3] = 1
            value[filledPos] = 1
            tmp.append(value)
        initialFactor.append(np.array(tmp))
    return initialFactor


def runNtfDemo(mu, sigma, eachSampleNum, initialValueUsed=False):
    # Generate samples as input data from Gaussians.
    x = generateVectorSample(eachSampleNum, mu, sigma)

    # Divide each axis to 5, 6, 7,...,(orders + 5).
    orders = np.array(mu).shape[1]
    baseBins = 5
    bins = np.arange(baseBins, baseBins + orders)
    hist, edge = transformSampleToHist(x, mu, sigma, bins)
    # showHistDistribution(hist, edge)

    # Use \# of Gaussians as \# of classes.
    classNum = len(mu)

    # Prepare factorization.
    ntfInstance = ntf.NTF(classNum, hist)

    # Set initial values.
    if initialValueUsed:
        initialFactor = generateInitialFactorValue(mu, edge, classNum)
        for i1, initial in enumerate(initialFactor):
            # print initial
            ntfInstance.setFactor(i1, initial)

    # Start factorization.
    ntfInstance.factorize(hist)

    # Normalize factors.
    # ntfInstance.normalizeFactor()

    # Show factors.
    showFactorValue(ntfInstance.getFactor())

    # Show reconstructed histogram from factors.
    reHist = ntfInstance.reconstruct()
    showHistDistribution(reHist, edge)
