#!/usr/bins/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import pandas as pd
from constant import *  # @UnusedWildImport
import time


def convertValueWithDictionary(dataFrame, column, dictionary):
    target = dataFrame[column].values
    target[:] = list(map(lambda x: dictionary[x], target))


def renameCouponDummy(couponDummy, valuePrefixed):
    return couponDummy


def renameCouponDetail(couponDetail, valuePrefixed):
    # Referred area when users bought coupon tickets.
    nameMap = {"SMALL_AREA_NAME": "REF_SMALL_AREA"}
    renamed = couponDetail.rename(columns=nameMap)
    if valuePrefixed:
        convertValueWithDictionary(renamed, "REF_SMALL_AREA", SMALL_AREA)
    return renamed


def renameCouponArea(couponArea, valuePrefixed):
    # Available area of coupon.
    nameMap = {"PREF_NAME": "ENABLE_PREF_NAME",
               "SMALL_AREA_NAME": "ENABLE_SMALL_AREA"}
    renamed = couponArea.rename(columns=nameMap)
    if valuePrefixed:
        convertValueWithDictionary(renamed, "ENABLE_PREF_NAME", PREFECTURE)
        convertValueWithDictionary(renamed, "ENABLE_SMALL_AREA", SMALL_AREA)
    return renamed


def renameUserList(userList, valuePrefixed):
    # Prefectures registered by users.
    nameMap = {"PREF_NAME": "USER_PREF_NAME"}
    renamed = userList.rename(columns=nameMap)
    if valuePrefixed:
        convertValueWithDictionary(renamed, "SEX_ID", SEX_TYPE)
        convertValueWithDictionary(renamed, "USER_PREF_NAME", PREFECTURE)
    return renamed


def renameCouponList(couponList, valuePrefixed):
    # Attributes of coupon.
    nameMap = {"large_area_name": "LIST_LARGE_AREA",
               "ken_name": "LIST_PREF_NAME",
               "small_area_name": "LIST_SMALL_AREA",
               "USABLE_DATE_MON": "USABLE_DATE 00 MON",
               "USABLE_DATE_TUE": "USABLE_DATE 01 TUE",
               "USABLE_DATE_WED": "USABLE_DATE 02 WED",
               "USABLE_DATE_THU": "USABLE_DATE 03 THU",
               "USABLE_DATE_FRI": "USABLE_DATE 04 FRI",
               "USABLE_DATE_SAT": "USABLE_DATE 05 SAT",
               "USABLE_DATE_SUN": "USABLE_DATE 06 SUN",
               "USABLE_DATE_HOLIDAY": "USABLE_DATE 07 HOLIDAY",
               "USABLE_DATE_BEFORE_HOLIDAY": "USABLE_DATE 08 BEFORE_HOLIDAY"
               }
    renamed = couponList.rename(columns=nameMap)
    if valuePrefixed:
        convertValueWithDictionary(renamed, "GENRE_NAME", GENRE)
        convertValueWithDictionary(renamed, "CAPSULE_TEXT", CAPSULE)
        convertValueWithDictionary(renamed, "LIST_PREF_NAME", PREFECTURE)
        convertValueWithDictionary(renamed, "LIST_SMALL_AREA", SMALL_AREA)
    return renamed


def readPonpareData(readingIndices=None, valuePrefixed=False):
    csvName = ["data/coupon_area_test.csv",     # 0
               "data/coupon_area_train.csv",    # 1
               "data/coupon_detail_train.csv",  # 2
               "data/coupon_list_test.csv",     # 3
               "data/coupon_list_train.csv",    # 4
               "data/coupon_visit_train.csv",   # 5
               "data/user_list.csv"]             # 6
    readingIndices = [0, 1, 2, 3, 4, 5, 6] \
        if None is readingIndices else readingIndices
    renameColumnFunc = [renameCouponArea,
                        renameCouponArea,
                        renameCouponDetail,
                        renameCouponList,
                        renameCouponList,
                        renameCouponDummy,
                        renameUserList]
    dataFrame = []
    for i1, renameColumn in enumerate(renameColumnFunc):
        if i1 not in readingIndices:
            continue
        name = csvName[i1]
        start = time.time()
        try:
            df = pd.read_csv(name)
        except IOError:
            print("===========================================================")
            print("Please download coupon data from:")
            print("- https://www.kaggle.com/c/coupon-purchase-prediction/data")
            print("===========================================================")
            raise
        dataFrame.append(renameColumn(df, valuePrefixed))
        print(" - %s -> elapsed time: %f[sec]" % (name, time.time() - start))
    return dataFrame
