#!/usr/bins/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import numpy as np


def generateFeatureLabelIf(label, bins):
    if label is None:
        labelBins = map(str, bins)
        labelBins.insert(0, "\"under\"")
        labelBins[-1] += " \"over\""
        label = ["%02d %s" % (i1, lb) for i1, lb in enumerate(labelBins)]
    return np.array(label)


def digitizeFeatureValue(feature, column, bins, label=None):
    value = feature[column].values
    label = generateFeatureLabelIf(label, bins)
    assert (len(bins) + 1) == len(label)
    feature[column] = label[np.digitize(value, np.array(bins))]
    return feature


def extractBinAndLabel(levels):
    bins = []
    label = []
    for i1 in levels:
        bins.append(np.arange(len(i1)))
        label.append(map(lambda x: str(x).decode('utf-8'), i1.values))
    return np.array(bins), np.array(label)


def createHistogram(dataFrame, extractColumn):
    group = dataFrame.groupby(extractColumn).size()
    index = group.index
    hist = np.zeros(map(len, index.levels))
    for i1, pos in enumerate(zip(*index.labels)):
        hist[pos] = group.values[i1]
    bins, label = extractBinAndLabel(index.levels)
    return hist, bins, label
