# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:55:51 2019
Anomaly detection with Gaussian Distribution/Recommender System with Collaborative Filtering
@author: 34276
"""
import numpy as np

def estimateGaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma

def selectThreshold(yval, pval):
    stepsize = (pval.max() - pval.min())/1000
    bestF1 = 0
    bestEps = 0
    for eps in np.arange(pval.min(), pval.max(), stepsize):
        preds = pval < eps
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)#1 if anomaly(positive), anomaly if pval<eps
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        p = tp/(tp+fp) 
        r = tp/(tp+fn)
        F1 = 2*p*r/(p+r)
        if F1 > bestF1:
            bestF1 = F1
            bestEps = eps
    #print(tp, fp, fn, p, r)#18, 570, 0, 0.03, 1
    return bestEps, bestF1

def cofiCostFunc(params, Y, R, num_features, lamb):
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features))
    X = np.matrix(X)#important
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))
    
    error = np.sum(np.power(np.multiply((X * Theta.T) - Y, R), 2))#相比于一般的cost function，这里需要考虑R
    regX = np.sum(np.power(X, 2))#collaborate filtering
    
    regTheta = np.sum(np.power(Theta, 2))
    J = 1/2*error + lamb/2*(regX+regTheta)
    
    X_grad = (np.multiply(X*Theta.T, R) - Y) * Theta + lamb * X
    Theta_grad = (np.multiply(X*Theta.T, R) - Y).T * X + lamb * Theta
    
    #grad = np.concatenate((X_grad, Theta_grad), axis=0)
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))#需要ravel
    return J,grad