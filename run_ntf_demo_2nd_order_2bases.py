#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/07/12

@author: drumichiro
'''
from ntf_demo_util import runNtfDemo


if __name__ == '__main__':
    # Generate samples as input data from Gaussians.
    mu = [[10, 20],
          [30, 30]]
    sigma = [[[5, 0],
              [0, 5]],
             [[5, 0],
              [0, 5]]]
    eachSampleNum = 100
    # Second order NTF is called NMF.
    runNtfDemo(mu, sigma, eachSampleNum)
    print("Done.")
