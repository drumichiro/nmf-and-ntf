#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/04/13

@author: drumichiro
'''
import numpy as np
import matplotlib.mlab as mlab


def generateSample(baseLength, mu, sigma, distFunc):
    x = np.empty([])
    np.random.seed(0)
    for i1 in range(len(mu)):
        data = distFunc(mu[i1], sigma[i1], baseLength)
        x = data if x.shape == () else np.append(x, data, axis=0)
    return x


def generateScalarSample(baseLength, mu, sigma):
    return generateSample(baseLength, mu, sigma, np.random.normal)


def generateVectorSample(baseLength, mu, sigma):
    return generateSample(baseLength, mu, sigma,
                          np.random.multivariate_normal)


def gaussian1d(x, mu, sigma):
    return mlab.normpdf(x, mu, sigma)


def gaussian2d(x, mu, sigma):
    return mlab.bivariate_normal(x[..., 0], x[..., 1],
                                 np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1]),
                                 mu[0], mu[1], sigma[0, 1])
