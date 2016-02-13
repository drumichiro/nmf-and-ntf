#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import numpy as np
import ntf
from myutil.sampler import generateVectorSample
from myutil.plotter import showFactorValue, showHistDistribution


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
    # Show source distribution.
    showHistDistribution(hist, edge=edge)

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
    showHistDistribution(reHist, edge=edge)
