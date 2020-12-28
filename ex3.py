# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:45:48 2019
用逻辑回归解决分类问题
@author: 34276
"""

import numpy as np

def sigmoid(z):
    return np.matrix(1/(1+np.exp(-z)))

def regCost(theta, X, y, lam):
    theta = np.matrix(theta).T
    #使用opt.fmin_tnc库时需要操作上一步！无论theta是什么shape，都会被压缩成array，所以第一步要reshape一下
    #print(np.shape(X), np.shape(theta))
    h = sigmoid(X*theta)
    m,n = np.shape(X)[0],np.shape(X)[1]
    theta1 = np.concatenate(([[0]], theta[1:n+1, :]), axis = 0)
    #print(np.shape(X), np.shape(h), np.shape(y), np.shape(theta), np.shape(theta1))
    return -1/m*(y.T*np.log(h)+(1-y).T*np.log(1-h))+lam/(2*m)*sum(np.power(theta1, 2))

def regGrad(theta, X, y, lam):
    theta = np.matrix(theta).T
    #使用opt.fmin_tnc库时需要操作上一步！无论theta是什么shape，都会被压缩成array，所以第一步要reshape一下
    h = sigmoid(X*theta)
    m,n = np.shape(X)[0],np.shape(X)[1]
    theta1 = np.concatenate(([[0]], theta[1:n+1, :]), axis = 0)
    #print(np.shape(X), np.shape(h), np.shape(y), np.shape(theta), np.shape(theta1))
    return 1/m*X.T*(h-y)+lam/m*theta1

from scipy.optimize import minimize
def oneVsAll(X, y, num_labels, lamb):
    n = np.shape(X)[1]
    initial_theta = np.matrix(np.zeros(n)).T
    all_theta = np.matrix(np.ones((n, num_labels)))
    for i in range(1, num_labels+1):
        y0 = np.matrix(np.array([1 if a==i else 0 for a in y[:,0]])).T
        result = minimize(regCost, initial_theta, args=(X, y0, lamb), method='TNC', jac=regGrad)#不注明tnc会报维度错误
        all_theta[:,i-1] = np.matrix(result.x).T
    return all_theta

def predictOneVsAll(all_theta, X):
    pMat = sigmoid(X*all_theta)
    return np.argmax(pMat, axis=1)+1

def accuracy(h, y):
    correct = np.array([1 if (a == b) else 0 for (a, b) in zip(h, y)])
    return sum(correct) / len(correct)