#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/02/13

@author: drumichiro
'''

import numpy as np
import pylab as plt
import matplotlib as mlib
import os.path as op
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

###########################################
EPS = 0.000000001
###########################################


def swapIndexForMeshGrid3d(array):
    if 2 == len(array):
        return np.copy(array[1]), np.copy(array[0])
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


def readFontProperty():
    # For Japanese output.
    fontPath = r"C:\WINDOWS\Fonts\YuGothic.ttf"
    if op.exists(fontPath):
        return FontProperties(fname=fontPath, size=14)
    return ""


def show2dDistribution(index, color, size, bins, label):
    i, j = index
    plt.figure(figsize=(16, 12))
    fp = readFontProperty()
    if label is not None:
        plt.xticks(bins[0], label[0], fontproperties=fp, rotation=90)
        plt.yticks(bins[1], label[1], fontproperties=fp)
    plt.scatter(i, j, color=color, s=size, marker="o")
    plt.show()


def show3dDistribution(index, color, size, bins, label):
    i, j, k = index
    fig = plt.figure(figsize=(16, 12))
    ax = Axes3D(fig)
    fp = readFontProperty()
    if label is not None:
        ax.set_xlim3d(-1, len(label[0]))
        ax.set_xticks(np.arange(len(label[0])))
        ax.set_xticklabels(label[0], fontproperties=fp,
                           horizontalalignment='left')
        ax.set_ylim3d(-1, len(label[1]))
        ax.set_yticks(np.arange(len(label[1])))
        ax.set_yticklabels(label[1], fontproperties=fp,
                           horizontalalignment='left')
        ax.set_zlim3d(-1, len(label[2]))
        ax.set_zticks(np.arange(len(label[2])))
        ax.set_zticklabels(label[2], fontproperties=fp,
                           horizontalalignment='left')
    ax.scatter3D(i, j, k, color=color, s=size, marker="o")
    plt.show()


def showDistribution(index, color, size, bins, label):
    dimention = len(index)
    if 2 == dimention:
        show2dDistribution(index, color, size, bins, label)
    elif 3 == dimention:
        show3dDistribution(index, color, size, bins, label)
    else:
        print("%dD scatter plot is unsupported." % dimention)


def showHistDistribution(hist, bins=None, edge=None, label=None):
    if None is bins:
        bins = list(map(np.arange, hist.shape))
    index = createIndexFromBin(bins)
    color = transformHistToColorStrength(hist)
    size = transformHistToSize(hist)
    showDistribution(index, color, size, bins, label)


def createIndexFromBin(bins):
    # This swap may not be necessary.
    swapedBins = swapIndexForMeshGrid3d(bins)
    grid = np.meshgrid(*swapedBins)
    index = np.array([i1.ravel() for i1 in grid])
    return swapIndexForMeshGrid3d(index)


def getMaxFactors(factor):
    maxFactors = []
    for fct1 in factor.T:
        tmp = []
        for fct2 in fct1:
            tmp.append(np.max(fct2))
        maxFactors.append(np.max(tmp))
    return maxFactors


def showFactorValue(factor):
    fig = plt.figure(figsize=(16, 12))
    fig.text(0.5, 0.04, 'Order', ha='center')
    fig.text(0.04, 0.5, 'Bases',
             va='center', rotation='vertical')
    colorList = ['b', 'g', 'r']
    colorLists = len(colorList)
    xLen, yLen = factor.shape[:2]
    elements = list(map(len, factor[0]))
    maxValue = getMaxFactors(factor)
    index = 0
    for i1 in np.arange(xLen):
        color = colorList[i1 % colorLists]
        for i2 in np.arange(yLen):
            index += 1
            ax = fig.add_subplot(xLen, yLen, index)
            ax.set_ylim([0, maxValue[i2]])
            line = np.arange(elements[i2])
            ax.bar(line, factor[i1][i2], color=color)
    plt.show()
