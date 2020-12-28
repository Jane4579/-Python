# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:50:46 2019
用神经网络解决分类问题
@author: 34276
"""

import numpy as np

def sigmoid(z):
    return np.matrix(1/(1+np.exp(-z)))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

def randInitializeWeights(L_in, L_out):
    IE = 0.1
    return np.random.rand(L_out, 1+L_in) * (2*IE) - IE

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h

#nn_params:压缩所有theta得到的向量
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    
    #theta1(hidden_layer_size, input_layer_size+1) theta2(num_labels, hidden_layer_size+1)
    theta1 = np.reshape(nn_params[:(input_layer_size+1)*hidden_layer_size], \
                                  (hidden_layer_size, input_layer_size+1))
    theta2 = np.reshape(nn_params[(input_layer_size+1)*hidden_layer_size:], \
                                  (num_labels, hidden_layer_size+1))
    m = np.shape(X)[0]
    
    #由y得到Y(m, num_labels)
    Y = np.matrix(np.zeros((m, num_labels)))
    for i in range(m):
        Y[i, y[i,:]-1] = 1;
        
    #fp
    a1 = np.insert(X, 0, 1, axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)#a2(m, 1+hidden_layer_size)
    z3 = a2 * theta2.T#a3(m, num_labels)
    h = sigmoid(z3)
    
    #compute J
    penalty = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))
    cost = 1/m*np.sum(np.multiply(-Y, np.log(h)) - np.multiply((1-Y), np.log(1-h)))
    J = cost + lamb/(2*m)*penalty
    #print(cost, J)
        
    #BP
    sigma3 = h-Y#(5000, 10)
    sigma2 = np.multiply(sigma3*theta2, sigmoidGradient(a2))
    sigma2 = sigma2[:, 1:]#sigma2(m, hidden_layer_size)
    
    #compute grad
    theta1_0 = np.insert(theta1[:, 1:], 0, 0, axis=1)#theta1_0(hidden_layer_size, input_layer_size+1)
    theta2_0 = np.insert(theta2[:, 1:], 0, 0, axis=1)
    theta1_grad = 1/m * (sigma2.T*a1 + lamb*theta1_0)
    theta2_grad = 1/m * (sigma3.T*a2 + lamb*theta2_0)
    grad = np.concatenate((np.ravel(theta1_grad), np.ravel(theta2_grad)))
    
    return J,grad