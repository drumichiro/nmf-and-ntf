#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2016/02/13

@author: drumichiro
'''

from myutil.histogram import *  # @UnusedWildImport
from myutil.ponpare.reader import *  # @UnusedWildImport
from myutil.ponpare.converter import *  # @UnusedWildImport
import pylab as plt
import ntf


def showLabel(label):
    for i1, lbl1 in enumerate(label):
        print "label:[%d] ->" % i1,
        for lbl2 in lbl1:
            print lbl2 + ",",
        print ""


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
    elements = map(len, factor[0])
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


def runNtfPonpareCoupon(column, bases):
    # Read coupon data.
    couponAreaTest, couponAreaTrain, couponDetailTrain, \
        couponListTest, couponListTrain, \
        couponVisitTrain, userList = readPonpareData(valuePrefixed=True)

    # Convert to one-hot expression.
    userList, couponListTrain, couponListTest = \
        digitizeHistoryFeatureValue(userList,
                                    couponListTrain,
                                    couponListTest)

    # Convert to histogram.
    distribution = transformForHistogram(userList,
                                         couponDetailTrain,
                                         couponVisitTrain,
                                         couponListTrain,
                                         couponListTest,
                                         couponAreaTrain,
                                         couponAreaTest)
    hist, bins, label = createHistogram(distribution, column)
    # showHistDistribution(hist, bins=bins, label=label)

    showLabel(label)
    # Start factorization.
    print "Start factorization..."
    ntfInstance = ntf.NTF(bases, hist, parallelCalc=True)
    ntfInstance.factorize(hist, showProgress=True)

    # Show factors
    # ntfInstance.normalizeFactor()
    print "Show value of factors..."
    showFactorValue(ntfInstance.getFactor())
