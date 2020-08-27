#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2016/02/13

@author: drumichiro
'''

from ntf_ponpare_coupon import runNtfPonpareCoupon


if __name__ == '__main__':
    runNtfPonpareCoupon(["GENRE_NAME", "SEX_ID", "LIST_PREF_NAME",
                         "AGE", "DISCOUNT_PRICE"], 8)
    print("Done.")
