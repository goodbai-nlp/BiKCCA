#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Kcca.py
@time: 17-4-7 下午8:41
"""
import numpy as np
from scipy.linalg import svd
from sklearn.metrics.pairwise import pairwise_kernels

class KCCA(object):
    """An implementation of Kernel Canonical Correlation Analysis.
    Based on code from Lorenzo Riano https://github.com/lorenzoriano/PyKCCA.
    """
    def __init__(self,kernel1, kernel2, regularization, ftype = 'full',
                 degree1=3,degree2=3,gamma1=None,gamma2=None, coef0=1, n_jobs=1,n_components = 20):
        if ftype not in ('full', 'icd'):
            raise ValueError("Error: valid decom values are full or icd, received: "+str(ftype))

        self.regular = regularization
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.coef0 = coef0
        self.n_jobs = n_jobs
        self.degree1 = degree1
        self.degree2 = degree2
        self.ftype = ftype
        self.n_components=n_components

    def fit(self,X,Y):
        self.trainX = X
        self.trainY = Y
        ndata_x, nfeature_x = X.shape
        ndata_y, nfeature_y = Y.shape
        if ndata_x != ndata_y:
            raise Exception("Inequality of number of data between X and Y")
        if self.ftype == "full":
            self.Kx = self._pairwise_kernels(self.kernel1,self.degree1,self.gamma1,X)
            self.Ky = self._pairwise_kernels(self.kernel2,self.degree2,self.gamma2,Y)
            (self.alpha, self.beta, self.corrs) = self.kcca(self.Kx, self.Ky)
            return self

    def _pairwise_kernels(self,ikernel, idegree,igamma, X, Y=None):
        return pairwise_kernels(X, Y,metric=ikernel, filter_params=True, n_jobs=self.n_jobs,
                                degree=idegree,
                                gamma=igamma, coef0=self.coef0)

    def kcca(self,Kx,Ky):
        '''
        计算投影向量 
        :param Kx: 核矩阵 Kx
        :param Ky: 核矩阵 Ky
        :return: alpha,beta,corrs
        '''
        I = self.regular * np.identity(self.Kx.shape[0])
        KxI_inv = np.linalg.inv(Kx + I)
        KyI_inv = np.linalg.inv(Ky + I)
        L = np.dot(KxI_inv, np.dot(Kx, np.dot(Ky, KyI_inv)))
        U, s, Vh = svd(L)

        self.alpha = np.dot(KxI_inv, U[:, :self.n_components])
        self.beta = np.dot(KyI_inv, Vh.T[:, :self.n_components])
        self.corrs = s[:self.n_components]
        return (self.alpha,self.beta,self.corrs)

    def transform(self, X1=None, X2=None):
        rets = []
        if X1 is not None:

            Ktest = self._pairwise_kernels(self.kernel1,self.degree1,self.gamma1,X1, self.trainX)
            res1 = np.dot(Ktest, self.alpha)
            rets.append(res1)

        if X2 is not None:

            Ktest = self._pairwise_kernels(self.kernel2,self.degree2,self.gamma1,X2, self.trainY)
            res2 = np.dot(Ktest, self.beta)
            rets.append(res2)
        return rets




