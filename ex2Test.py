# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:55:35 2019

@author: 34276
"""

import ex2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#获取数据
path =  r"C:\Users\34276\Documents\machine learning\machine-learning-ex2\ex2\ex2data1.txt"
data = pd.read_csv(path, names=['Exam 1 score', 'Exam 2 score', 'Admitted'])
path2 =  r"C:\Users\34276\Documents\machine learning\machine-learning-ex2\ex2\ex2data2.txt"
data2 = pd.read_csv(path2, names=['Test 1', 'Test 2', 'Accepted'])

#绘制分类点
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(positive['Test 1'], positive['Test 2'], marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], marker='+', label='Not Accepted')

#得到X和y
data2.insert(0, 'ones', 1)
dataMat = np.matrix(data2)
X,y = dataMat[:, 0:3],dataMat[:, 3]
theta = np.matrix([[0], [0], [0]])

#计算J和grad
J,grad = ex2.regCostGrad(X, y, theta, 1)

#用SciPy's truncated newton（TNC）实现寻找最优参数,利用得到的最优参数进行预测，并计算准确性
import scipy.optimize as opt
result = opt.fmin_tnc(func=ex2.regCostGrad, x0=theta, args=(X, y, 1))
theta = np.matrix(result[0]).T
h = ex2.predict(theta, X)
accuracy = ex2.accuracy(h, y)