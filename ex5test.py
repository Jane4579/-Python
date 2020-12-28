# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:12:23 2019
通过learning curve判断迭代效果，通过validation curve寻找最优lambda
@author: 34276
"""
import ex5
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('ex5data1.mat')
X, y, Xtest, ytest, Xval, yval = data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data['yval']
X, Xval, Xtest = np.insert(X, 0, 1, axis=1), np.insert(Xval, 0, 1, axis=1), np.insert(Xtest, 0, 1, axis=1)

#使用训练集和验证集直接汇总learning curve，lamb为0
J_train, J_val = ex5.learningCurve(X, y, Xval, yval, 0)
ex5.plotLC(J_train, J_val)#J_train较大，欠拟合

#对训练集和验证集都进行多项式化之后绘制learning curve，lamb为0
X_poly = ex5.polyFeatures(X, 8)
Xval_poly = ex5.polyFeatures(Xval, 8)
Xtest_poly = ex5.polyFeatures(Xtest, 8)
J_train, J_val = ex5.learningCurve(X_poly, y, Xval_poly, yval, 0)
ex5.plotLC(J_train, J_val)#J_train几乎为0，过拟合

#通过绘制validation curve得到最优lamb
lambda_vec, J_train, J_val = ex5.validationCurve(X_poly, y, Xval_poly, yval)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(lambda_vec, J_train, label='Traning Data')
ax.plot(lambda_vec, J_val, label='Validation Data')
ax.set_xlabel('lamda')
ax.set_ylabel('error')
ax.set_title('Validation Curves')
plt.legend(loc='upper right')

#使用最优lamb绘制learning curve
best_lamb = lambda_vec[np.argmin(J_val)]
J_train, J_val = ex5.learningCurve(X_poly, y, Xval_poly, yval, best_lamb)
ex5.plotLC(J_train, J_val)#J_train较小，拟合效果较好