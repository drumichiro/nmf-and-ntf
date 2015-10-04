#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/10/04

@author: drumichiro
'''

import numpy as np
import unittest
import ntf
from myutil.sampler import generateVectorSample
from ntf_demo_util import transformSampleToHist


class NtfTest(unittest.TestCase):
#     def setUp(self):
#         print "setUp"
# 
#     def tearDown(self):
#         print "tearDown"

    def generateGaussianHistgram(self, mu, sigma, eachSampleNum):
        # Generate samples as input data from Gaussians.
        x = generateVectorSample(eachSampleNum, mu, sigma)
        hist, _ = transformSampleToHist(x, mu, sigma)
        return hist

    def evaluateNtfReconstruction(self, mu, sigma, eachSampleNum, clussNum):
        srcHist = self.generateGaussianHistgram(mu, sigma, eachSampleNum)

        # Do NTF.
        ntfInstance = ntf.NTF(clussNum, srcHist)
        ntfInstance.factorize(srcHist)
        dstHist = ntfInstance.reconstruct()

        # Calculate a difference between source and destination histogram.
        diffHist = srcHist - dstHist
        diffHistSum = np.sum(diffHist*diffHist)
        srcHistSum = np.sum(srcHist*srcHist)
        return 1.0 - diffHistSum/srcHistSum

    def testNtf2ndOrder(self):
        # Generate used samples.
        mu = [[10, 20],
              [30, 30]]
        sigma = [[[5, 0],
                  [0, 5]],
                 [[5, 0],
                  [0, 5]]]
        eachSampleNum = 100
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma,
                                                  eachSampleNum, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma,
                                                  eachSampleNum, len(mu))
        self.assertGreater(accuracy, 0.9995)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma,
                                                 eachSampleNum, len(mu) + 1)
        self.assertGreater(overfit, accuracy)

    def testNtf3rdOrder(self):
        # Generate used samples.
        mu = [[10, 10, 20],
              [30, 30, 30]]
        sigma = [[[5, 0, 0],
                  [0, 5, 0],
                  [0, 0, 5]],
                 [[5, 0, 0],
                  [0, 5, 0],
                  [0, 0, 5]]]
        eachSampleNum = 100
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma,
                                                  eachSampleNum, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma,
                                                  eachSampleNum, len(mu))
        self.assertGreater(accuracy, 0.995)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma,
                                                 eachSampleNum, len(mu) + 1)
        self.assertGreater(overfit, accuracy)

    def testNtf4thOrder(self):
        # Generate used samples.
        mu = [[10, 10, 20, 20],
              [30, 30, 30, 30]]
        sigma = [[[5, 0, 0, 0],
                  [0, 5, 0, 0],
                  [0, 0, 5, 0],
                  [0, 0, 0, 5]],
                 [[5, 0, 0, 0],
                  [0, 5, 0, 0],
                  [0, 0, 5, 0],
                  [0, 0, 0, 5]]]
        eachSampleNum = 100
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma,
                                                  eachSampleNum, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma,
                                                  eachSampleNum, len(mu))
        self.assertGreater(accuracy, 0.985)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma,
                                                 eachSampleNum, len(mu) + 1)
        self.assertGreater(overfit, accuracy)
