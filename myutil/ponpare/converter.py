#!/usr/bins/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import pandas as pd
from ..histogram import digitizeFeatureValue


def digitizeHistoryFeatureValue(userList,
                                couponListTrain, couponListTest):
    ageBins = [20, 25, 30, 35, 40, 45,
               50, 55, 60, 65, 70, 75]
    userList = digitizeFeatureValue(userList, "AGE", ageBins)

    priceRateBins = [50, 70, 80, 90, 100]
    couponListTrain = digitizeFeatureValue(couponListTrain,
                                           "PRICE_RATE", priceRateBins)
    couponListTest = digitizeFeatureValue(couponListTest,
                                          "PRICE_RATE", priceRateBins)

    discountBins = [100, 1000, 2000, 3000, 5000,
                    10000, 20000, 30000, 50000]
#     discountBins = [100, 500, 1000, 1500, 2000, 2500,
#                     3000, 4000, 5000, 6000, 7000, 8000, 9000,
#                     10000, 20000, 30000, 50000]
    couponListTrain = digitizeFeatureValue(couponListTrain,
                                           "DISCOUNT_PRICE", discountBins)
    couponListTest = digitizeFeatureValue(couponListTest,
                                          "DISCOUNT_PRICE", discountBins)

    validsBins = [10, 30, 60, 90, 120, 150]
    couponListTrain = digitizeFeatureValue(couponListTrain,
                                           "VALIDPERIOD", validsBins)
    couponListTest = digitizeFeatureValue(couponListTest,
                                          "VALIDPERIOD", validsBins)

    return userList, couponListTrain, couponListTest


def transformForHistogram(userList, couponDetailTrain, couponVisitTrain,
                          couponListTrain, couponListTest,
                          couponAreaTrain, couponAreaTest):

    visit = pd.merge(couponListTest, couponVisitTrain)
    visit = visit[["COUPON_ID_hash", "USER_ID_hash"]].drop_duplicates()
    history = pd.merge(couponDetailTrain, visit,
                       on=["USER_ID_hash", "COUPON_ID_hash"], how="outer")
    history["ITEM_COUNT"] = history["ITEM_COUNT"].fillna(0)
    history["REF_SMALL_AREA"] = history["REF_SMALL_AREA"].fillna("無登録")
    couponList = pd.concat([couponListTrain, couponListTest])
    history = pd.merge(history, couponList,
                       on="COUPON_ID_hash", how="left")

    assert len(history) > len(couponListTrain)

    # Deal with a purchaser as an attribute of coupon.
    # The users which purchase no coupon become NaN.
    history = pd.merge(userList, history,
                       on="USER_ID_hash", how="left")
    assert len(history) > len(couponListTrain)
    assert len(history) > len(userList)
    return history
