# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:30:13 2019
K-means&PCA
@author: 34276
"""
import numpy as np

def findClosestCentroids(X, centroids):
    m = np.shape(X)[0]
    index = np.zeros((m))
    K = np.shape(centroids)[0]
    for i in range(m):
        min_dist = 1000000
        for j in range(K):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                index[i] = j        
    return index

def computeCentroids(X, idx, K):
    m,n = np.shape(X)
    s = np.zeros((K, n))
    count = np.zeros((K, 1))
    centroids = np.zeros((K, n))
    for j in range(K):  
        for i in range(m):
            if idx[i] == j:
                s[j,:] = s[j,:] + X[i, :]
                count[j] = count[j] + 1
        centroids[j, :] = s[j, :]/count[j]
    return centroids

def kMeansInitCentroids(X, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    idx = np.random.randint(0, m, K)#随机取0到m之间的3个整数
    #print(idx)
    for i in range(K):
        centroids[i,:] = X[idx[i],:]#随机取X中的3个样本点作为初始中心
    return centroids

def runkMeans(X, centroids, max_iters):
    K = np.shape(centroids)[0]
    for i in range(max_iters):
        index = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, index, K)
    return index,centroids

def pca(X):
    X = (X - X.mean()) / X.std()
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    return U, S, V

def projectData(X, U, K):
    U_reduced = U[:,:K]
    return X * U_reduced
       
def recoverData(Z, U, K):
    U_reduced = U[:,:K]
    return Z * U_reduced.T