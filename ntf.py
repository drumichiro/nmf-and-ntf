#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import numpy as np


###########################################
EPS = 0.0000001
###########################################


class NTF():
    def __init__(self, bases, x, costFuncType='euclid'):
        self.shape = x.shape
        self.factor = self.allocateFactor(bases)
        # Preset shape to be easy for broadcast.
        dimention = len(self.shape)
        self.preshape = np.tile(self.shape, dimention).reshape(dimention, -1)
        for i1 in np.arange(dimention):
            self.preshape[i1, i1] = 1
        # Select update rule based on a cost function.
        if 'euclid' == costFuncType:
            self.updater = self.updateBasedOnEuclid
        elif 'gkld' == costFuncType:
            self.updater = self.updateBasedOnGKLD
        elif 'isd' == costFuncType:
            self.updater = self.updateBasedOnISD
        else:
            assert False, "\"" + costFuncType + "\" is invalid."

    def allocateFactor(self, bases):
        factor = []
        for _ in np.arange(bases):
            tmp = []
            for i2 in self.shape:
                tmp.append(np.ones(i2))
            factor.append(tmp)
        return np.array(factor)

    def sumAlongIndex(self, value, factor, index):
        for _ in np.arange(index):
            value = np.sum(value, axis=0)
        for _ in np.arange(index + 1, len(factor)):
            value = np.sum(value, axis=1)
        return value

    def kronAlongIndex(self, factor, index):
        element = np.array([1])
        for i1 in factor[:index]:
            element = np.kron(element, i1)
        for i1 in factor[index + 1:]:
            element = np.kron(element, i1)
        return element

    def createTensorFromFactors(self):
        tensor = 0
        for i1 in self.factor:
            tensor += self.kronAlongIndex(i1, len(i1))
        return tensor.reshape(self.shape)

    def updateBasedOnEuclid(self, x, factor, index):
        # Create tensor partly.
        element = self.kronAlongIndex(factor, index)

        # Summation
        element = element.reshape(self.preshape[index])
        numer = self.sumAlongIndex(x*element, factor, index)
        estimation = self.createTensorFromFactors()
        denom = self.sumAlongIndex(estimation*element, factor, index)

        return numer/(denom + EPS)

    def updateBasedOnGKLD(self, x, factor, index):
        # Create tensor partly.
        element = self.kronAlongIndex(factor, index)

        # Summation
        element = element.reshape(self.preshape[index])
        estimation = self.createTensorFromFactors()
        boost = x/(estimation + EPS)
        numer = self.sumAlongIndex(boost*element, factor, index)
        denom = np.sum(element)

        return numer/(denom + EPS)

    def updateBasedOnISD(self, x, factor, index):
        # TODO: implement this.
        assert False, "This cost function is unsupported now."
        return 0

    def updateFactorEachBasis(self, x, factorPerBasis):
        for i1 in np.arange(len(factorPerBasis)):
            factorPerBasis[i1] *= self.updater(x, factorPerBasis, i1)

    def updateAllFactors(self, x, factor):
        for i1 in factor:
            self.updateFactorEachBasis(x, i1)

    def factorize(self, x, iterations=100):
        for _ in np.arange(iterations):
            self.updateAllFactors(x, self.factor)

    def reconstruct(self):
        return self.createTensorFromFactors()

    def setFactor(self, dimention, initialValue):
        assert len(initialValue) == len(self.factor)
        assert dimention < len(self.factor[0])
        assert initialValue.shape[1] == len(self.factor[0][dimention])
        for i1, value in enumerate(initialValue):
            self.factor[i1][dimention] = value + EPS

    def getFactor(self):
        return np.copy(self.factor)

    def getNormalizedFactor(self):
        weight = []
        normalized = []
        for i1 in self.factor:
            baseValue = np.sum(i1, axis=1)
            weight = np.append(weight, np.prod(baseValue))
            normalized = np.append(normalized, i1/baseValue.reshape(-1, 1))
        return weight, normalized.reshape(self.factor.shape)


# For easy unit test
if __name__ == '__main__':
    # test = np.arange(60).reshape(3, 4, 5)
    test = np.arange(24).reshape(2, 3, 4)
    # test = np.arange(6).reshape(1, 2, 3)
    ntf = NTF(1, test)
    ntf.factorize(test)
    print ntf.reconstruct()
