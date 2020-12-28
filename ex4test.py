# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:14:15 2019

@author: 34276
"""
import ex4
import numpy as np
from scipy.io import loadmat

data = loadmat('ex4data1.mat')
X = np.matrix(data['X'])
y = np.matrix(data['y'])

input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
params = (np.random.random(size=hidden_size * (input_size + 1) + \
                           num_labels * (hidden_size + 1)) - 0.5) * 0.25
J,grad = ex4.nnCostFunction(params, input_size, hidden_size, num_labels, X, y, learning_rate)

from scipy.optimize import minimize

# minimize the objective function
fmin = minimize(fun=ex4.nnCostFunction, x0=params, args=(input_size, hidden_size, num_labels, X, y, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 250})

theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = ex4.forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))