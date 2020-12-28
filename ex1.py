# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:46:18 2019

@author: 34276
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

def computeCost(X, y, theta):
    error = y-X*theta
    return sum(np.power(error, 2))/(2*len(y))

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters);
    for i in range(num_iters):
        h = X * theta;
        theta = theta - alpha/len(y)*X.T*(h-y);
        J_history[i] = computeCost(X, y, theta);
    return theta, J_history

def normalEqn(X, y):
    return (X.T*X).I*X.T*y