#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2016/02/13

@author: drumichiro
'''

import ntf
from myutil.histogram import createHistogram
from myutil.plotter import showFactorValue, showHistDistribution
from myutil.ponpare.reader import readPonpareData
from myutil.ponpare.converter import \
    digitizeHistoryFeatureValue, transformForHistogram


def showLabel(label):
    for i1, lbl1 in enumerate(label):
        print("label:[%d] ->" % i1, end="")
        for lbl2 in lbl1:
            print(lbl2 + ",", end="")
        print("")


def runNtfPonpareCoupon(column, bases):
    # Read coupon data.
    couponAreaTest, couponAreaTrain, couponDetailTrain, \
        couponListTest, couponListTrain, \
        couponVisitTrain, userList = readPonpareData(valuePrefixed=True)

    # WAR: Fix the wrong column in coupon data.
    couponVisitTrain = couponVisitTrain.rename(columns={'VIEW_COUPON_ID_hash':
                                                        'COUPON_ID_hash'})

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
    showHistDistribution(hist, bins=bins, label=label)

    showLabel(label)
    # Start factorization.
    print("Start factorization...")
    ntfInstance = ntf.NTF(bases, hist, parallelCalc=True)
    ntfInstance.factorize(hist, showProgress=True)

    # Show factors
    # ntfInstance.normalizeFactor()
    print("Show value of factors...")
    showFactorValue(ntfInstance.getFactor())
