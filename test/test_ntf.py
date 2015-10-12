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
from ntf_demo_util import generateInitialFactorValue


class NtfTest(unittest.TestCase):

    def generateGaussianHistgram(self, mu, sigma, eachSamples):
        # Generate samples as input data from Gaussians.
        x = generateVectorSample(eachSamples, mu, sigma)

        # Divide each axis to 5, 6, 7,...,(orders + 5).
        orders = np.array(mu).shape[1]
        baseBins = 5
        bins = np.arange(baseBins, baseBins + orders)
        hist, edge = transformSampleToHist(x, mu, sigma, bins)
        return hist, edge

    def evaluateNtfReconstruction(self, mu, sigma, eachSamples,
                                  costFuncType, classNum):
        srcHist, edge = self.generateGaussianHistgram(mu, sigma, eachSamples)

        # Prepare for NTF.
        ntfInstance = ntf.NTF(classNum, srcHist, costFuncType)

        # Set initial values.
        # print "--------------------"
        initialFactor = generateInitialFactorValue(mu, edge, classNum)
        for i1, initial in enumerate(initialFactor):
            # print initial
            ntfInstance.setFactor(i1, initial)

        # Do NTF.
        ntfInstance.factorize(srcHist)
        dstHist = ntfInstance.reconstruct()

#         print "===================="
#         for factor in ntfInstance.getFactor():
#             print factor

        # Calculate a difference between source and destination histogram.
        diffHist = srcHist - dstHist
        diffHistSum = np.sum(diffHist*diffHist)
        srcHistSum = np.sum(srcHist*srcHist)
        return 1.0 - diffHistSum/srcHistSum

    def get2ndOrderGaussianParameter(self):
        mu = [[10, 20],
              [30, 30]]
        sigma = [[[5, 0],
                  [0, 5]],
                 [[5, 0],
                  [0, 5]]]
        return mu, sigma

    def get3rdOrderGaussianParameter(self):
        # Generate used samples.
        mu = [[10, 10, 20],
              [30, 30, 30]]
        sigma = [[[5, 0, 0],
                  [0, 5, 0],
                  [0, 0, 5]],
                 [[5, 0, 0],
                  [0, 5, 0],
                  [0, 0, 5]]]
        return mu, sigma

    def get4thOrderGaussianParameter(self):
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
        return mu, sigma

    def get5thOrderGaussianParameter(self):
        mu = [[10, 10, 20, 20, 20],
              [30, 30, 30, 30, 30]]
        sigma = [[[5, 0, 0, 0, 0],
                  [0, 5, 0, 0, 0],
                  [0, 0, 5, 0, 0],
                  [0, 0, 0, 5, 0],
                  [0, 0, 0, 0, 5]],
                 [[5, 0, 0, 0, 0],
                  [0, 5, 0, 0, 0],
                  [0, 0, 5, 0, 0],
                  [0, 0, 0, 5, 0],
                  [0, 0, 0, 0, 5]]]
#         mu = [[10, 10, 20, 20, 10, 10],
#               [30, 30, 30, 30, 30, 30],
#               [30, 30, 30, 30, 30, 30]]
#         sigma = [[[5, 0, 0, 0, 0, 0],
#                   [0, 5, 0, 0, 0, 0],
#                   [0, 0, 5, 0, 0, 0],
#                   [0, 0, 0, 5, 0, 0],
#                   [0, 0, 0, 0, 5, 0],
#                   [0, 0, 0, 0, 0, 5]],
#                  [[5, 0, 0, 0, 0, 0],
#                   [0, 5, 0, 0, 0, 0],
#                   [0, 0, 5, 0, 0, 0],
#                   [0, 0, 0, 5, 0, 0],
#                   [0, 0, 0, 0, 5, 0],
#                   [0, 0, 0, 0, 0, 5]],
#                  [[5, 0, 0, 0, 0, 0],
#                   [0, 5, 0, 0, 0, 0],
#                   [0, 0, 5, 0, 0, 0],
#                   [0, 0, 0, 5, 0, 0],
#                   [0, 0, 0, 0, 5, 0],
#                   [0, 0, 0, 0, 0, 5]]]
        return mu, sigma

    def testNtf2ndOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get2ndOrderGaussianParameter()
        eachSamples = 100
        costFuncType = "euclid"
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu))
        self.assertGreater(accuracy, 0.98)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                 costFuncType, len(mu) + 1)
        self.assertGreater(overfit, accuracy)

    def testNtf3rdOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get3rdOrderGaussianParameter()
        eachSamples = 100
        costFuncType = "euclid"
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu))
        self.assertGreater(accuracy, 0.98)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                 costFuncType, len(mu) + 1)
        self.assertGreater(overfit, accuracy)

    def testNtf4thOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get4thOrderGaussianParameter()
        eachSamples = 100
        costFuncType = "euclid"
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu))
        self.assertGreater(accuracy, 0.95)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                 costFuncType, len(mu) + 1)
        self.assertGreater(overfit, accuracy)

    def testNtf5thOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get5thOrderGaussianParameter()
        eachSamples = 100
        costFuncType = "euclid"
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu) - 1)
        self.assertLess(accuracy, 0.7)
        # Check a basic case.
        accuracy = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                  costFuncType, len(mu))
        self.assertGreater(accuracy, 0.85)
        # Check a case of over fitting.
        overfit = self.evaluateNtfReconstruction(mu, sigma, eachSamples,
                                                 costFuncType, len(mu) + 1)
        self.assertGreater(overfit, accuracy)
