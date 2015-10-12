#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/06/14

@author: drumichiro
'''

import numpy as np
from multiprocessing import Pool


###########################################
EPS = 0.0000001
###########################################


class MulHelper(object):
    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)


class NTF():
    def __init__(self, bases, x, costFuncType='euclid', parallelCalc=False):
        self.shape = x.shape
        self.factor = self.allocateFactor(bases)
        # Preset shape to be easy for broadcast.
        dimention = len(self.shape)
        self.preshape = np.tile(self.shape, dimention).reshape(dimention, -1)
        for i1 in np.arange(dimention):
            self.preshape[i1, i1] = 1

        if parallelCalc:
            self.pool = Pool()
            self.composeTensor = self.composeTensorParallely
        else:
            self.composeTensor = self.composeTensorSerially

        # Select update rule based on a cost function.
        if 'euclid' == costFuncType:
            self.updater = self.updateBasedOnEuclid
        elif 'gkld' == costFuncType:
            self.updater = self.updateBasedOnGKLD
        elif 'isd' == costFuncType:
            self.updater = self.updateBasedOnISD
        else:
            assert False, "\"" + costFuncType + "\" is invalid."

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']

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

    def composeTensorSerially(self, element):
        return map(self.kronAll, element)

    def composeTensorParallely(self, element):
        return self.pool.map(MulHelper(self, 'kronAll'), element)

    def kronAll(self, factor):
        element = np.array([1])
        for i1 in factor:
            element = np.kron(element, i1)
        return element

    def kronAlongIndex(self, factor, index):
        element = np.array([1])
        for i1 in factor[:index]:
            element = np.kron(element, i1)
        for i1 in factor[index + 1:]:
            element = np.kron(element, i1)
        return element

    def createTensorFromFactors(self):
        tensor = self.composeTensor(self.factor)
        tensor = np.sum(tensor, axis=0)
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

    def factorize(self, x, iterations=100, showProgress=False):
        for i1 in np.arange(1, iterations + 1):
            if showProgress:
                progress = "*" if 0 < (i1 % 20) \
                    else "[%d/%d]\n" % (i1, iterations)
                print progress,
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
        for fct in self.factor:
            baseValue = np.empty(len(fct))
            for i1 in np.arange(len(fct)):
                baseValue[i1] = np.sum(fct[i1])
            weight = np.append(weight, np.prod(baseValue))
            tmp = []
            for fct2, base in zip(fct, baseValue):
                tmp.append(fct2/base)
            normalized.append(tmp)
        return weight, np.array(normalized)


# For easy unit test
if __name__ == '__main__':
    # test = np.arange(60).reshape(3, 4, 5)
    test = np.arange(24).reshape(2, 3, 4)
    # test = np.arange(6).reshape(1, 2, 3)
    ntf = NTF(1, test)
    ntf.factorize(test)
    print ntf.reconstruct()
