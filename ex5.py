# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:44:53 2019

@author: 34276
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def linearRegCostFunction(theta, X, y, lamb):
    m = np.shape(X)[0]
    theta = np.matrix(theta)
    h = X * theta.T
    theta0 = np.insert(theta[:, 1:], 0, 0, axis=1)
    #print(np.shape(h-y), np.shape(theta0))
    J = 1/(2*m) * ((h-y).T*(h-y) + lamb*theta0*theta0.T)
    grad = 1/m * (X.T*(h-y) + lamb*theta0.T)    
    return J,grad

def learningCurve(X, y, Xval, yval, lamb):
    m,n = np.shape(X)
    theta = np.matrix(np.zeros((n)))
    J_train,J_val = np.zeros((m)),np.zeros((m))
    for i in range(m):
        fmin = minimize(fun=linearRegCostFunction, x0=theta, args=(X[0:i+1,:], y[0:i+1,:], lamb), method='TNC', jac=True)
        theta = fmin['x']
        J_train[i] = linearRegCostFunction(theta, X[0:i+1,:], y[0:i+1,:], 0)[0]
        J_val[i] = linearRegCostFunction(theta, Xval, yval, 0)[0]
    return J_train, J_val

def plotLC(J_train, J_val):
    x = range(1, 13)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, J_train, label='Traning Data')
    ax.plot(x, J_val, label='Validation Data')
    ax.set_xlabel('m')
    ax.set_ylabel('error')
    ax.set_title('Learning Curves')
    plt.legend(loc='upper right')

def polyFeatures(X, p):#X(m,2)
    X = X[:, 1:]
    X_poly = np.matrix(np.ones((np.shape(X)[0], 1)))
    for i in range(1, p+1):
        X_poly = np.concatenate((X_poly, np.power(X, i)), axis=1) 
    X_poly = X_poly[:, 1:]    
    for i in range(np.shape(X_poly)[1]):
        X_poly[:,i] = (X_poly[:,i] - X_poly[:,i].mean()) / X_poly[:,i].std()
    #X_poly = (X_poly - X_poly.mean()) / X_poly.std()
    X_poly = np.insert(X_poly, 0, 1, axis=1)
    return X_poly

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    l = len(lambda_vec)
    J_train,J_val = np.zeros((l)),np.zeros((l))
    theta = np.matrix(np.zeros((np.shape(X)[1])))
    for i in range(l):
        fmin = minimize(fun=linearRegCostFunction, x0=theta, args=(X, y, lambda_vec[i]), method='TNC', jac=True)
        theta = fmin['x']
        J_train[i] = linearRegCostFunction(theta, X, y, 0)[0]
        J_val[i] = linearRegCostFunction(theta, Xval, yval, 0)[0]
        print(i, lambda_vec[i], theta[0], J_train[i], J_val[i])
    return lambda_vec, J_train, J_val