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

    def generateGaussianHistogram(self, mu, sigma, eachSamples):
        # Generate samples as input data from Gaussians.
        x = generateVectorSample(eachSamples, mu, sigma)

        # Divide each axis to 5, 6, 7,...,(orders + 5).
        orders = np.array(mu).shape[1]
        baseBins = 5
        bins = np.arange(baseBins, baseBins + orders)
        hist, edge = transformSampleToHist(x, mu, sigma, bins)
        return hist, edge

    def calculateHistAccuracy(self, srcHist, dstHist):
        diffHist = srcHist - dstHist
        diffHistSum = np.sum(diffHist*diffHist)
        srcHistSum = np.sum(srcHist*srcHist)
        return 1.0 - diffHistSum/srcHistSum

    def evaluateNtfReconst(self, mu, sigma, eachSamples,
                           costFuncType, classNum):
        srcHist, edge = self.generateGaussianHistogram(mu, sigma, eachSamples)

        # Prepare for NTF.
        ntfInstance = ntf.NTF(classNum, srcHist, costFuncType)

        # Set initial values.
        initialFactor = generateInitialFactorValue(mu, edge, classNum)
        for i1, initial in enumerate(initialFactor):
            # print initial
            ntfInstance.setFactor(i1, initial)

        # Do NTF.
        ntfInstance.factorize(srcHist)
        dstHist = ntfInstance.reconstruct()

        # Calculate a difference between source and destination histogram.
        return self.calculateHistAccuracy(srcHist, dstHist)

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
        return mu, sigma

    def calculateAccuracy(self, mu, sigma, costFuncType):
        eachSamples = 100
        # Check a case using lacking basis.
        accuracy = self.evaluateNtfReconst(mu, sigma, eachSamples,
                                           costFuncType, len(mu) - 1)
        print("insufficient: %f" % (accuracy))
        insufficientAccuracy = accuracy
        # Check a basic case.
        accuracy = self.evaluateNtfReconst(mu, sigma, eachSamples,
                                           costFuncType, len(mu))
        print("sufficient  : %f" % (accuracy))
        sufficientAccuracy = accuracy
        # Check a case of over fitting.
        accuracy = self.evaluateNtfReconst(mu, sigma, eachSamples,
                                           costFuncType, len(mu) + 1)
        print("overfit     : %f" % (accuracy))
        overfitAccuracy = accuracy
        return insufficientAccuracy, sufficientAccuracy, overfitAccuracy

    def testNtf2ndOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get2ndOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "euclid")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.992)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf2ndOrderWithGkld(self):
        # Generate used samples.
        mu, sigma = self.get2ndOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "gkld")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.989)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf3rdOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get3rdOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "euclid")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.995)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf3rdOrderWithGkld(self):
        # Generate used samples.
        mu, sigma = self.get3rdOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "gkld")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.991)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf4thOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get4thOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "euclid")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.962)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf4thOrderWithGkld(self):
        # Generate used samples.
        mu, sigma = self.get4thOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "gkld")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.952)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf5thOrderWithEuclid(self):
        # Generate used samples.
        mu, sigma = self.get5thOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "euclid")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.851)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)

    def testNtf5thOrderWithGkld(self):
        # Generate used samples.
        mu, sigma = self.get5thOrderGaussianParameter()
        insufficientAccuracy, sufficientAccuracy, overfitAccuracy \
            = self.calculateAccuracy(mu, sigma, "gkld")

        self.assertLess(insufficientAccuracy, 0.7)
        self.assertGreater(sufficientAccuracy, 0.821)
        leftBoudaryAccuracy = sufficientAccuracy - 0.02
        rightBoudaryAccuracy = sufficientAccuracy - 0.01
        self.assertLess(leftBoudaryAccuracy, overfitAccuracy)
        self.assertGreater(overfitAccuracy, rightBoudaryAccuracy)
