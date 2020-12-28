# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:53:04 2019

@author: 34276
"""

import numpy as np
#import matplotlib.pyplot as plt

def sigmoid(z):
    return np.matrix(1/(1+np.exp(-z)))

def costGrad(X, y, theta):
    m = len(y)
    h = sigmoid(X*theta)
    J = -1/m*(y.T*np.log(h)+(1-y).T*np.log(1-h))
    grad = 1/m*X.T*(h-y)
    return J, grad

def predict(theta, X):
    m = len(X)
    h = np.zeros(m)
    for i in range(m):
        ht = sigmoid(X[i, :]*theta);
        if ht>=0.5: h[i] = 1;
        else: h[i] = 0;
    return h;

def accuracy(h, y):
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(h, y)]
    return (sum(correct)) / len(correct)

def regCostGrad(theta, X, y, lam):
    theta = np.matrix(theta).T#使用opt.fmin_tnc库时需要操作这一步！无论theta是什么shape，都会被压缩成array，所以第一步要reshape一下
    #print(np.shape(X), np.shape(theta))
    h = sigmoid(X*theta)
    m,n = len(y),len(theta)
    theta1 = np.concatenate(([[0]], theta[1:n+1, :]), axis = 0)
    J = -1/m*(y.T*np.log(h)+(1-y).T*np.log(1-h))+lam/(2*m)*sum(np.power(theta1, 2))
    grad = 1/m*X.T*(h-y)+lam/m*theta1
    return J, grad

def mapFeature(X, degree):#X为二列矩阵    
    X = np.matrix(X)
    X1,X2 = X[:,0],X[:,1]
    out = np.matrix(np.ones(len(X))).T
    for i in range(1, degree+1):#不考虑degree为0的情况
        for j in range(i+1):
            out = np.append(out, np.multiply(np.power(X1,(i-j)),np.power(X2,j)), axis=1);
            #print(i, j, out)
    return out