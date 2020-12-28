# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:29:44 2019

@author: 34276
"""

import ex1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.matrix(eye(5))

path =  r"C:\Users\34276\Documents\machine learning\machine-learning-ex1\ex1\ex1data1.txt"
data = pd.read_csv(path, names=['Population', 'Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
data.insert(0, 'ones', 1)
dataMat = matrix(data)
X, y = dataMat[:, 0:2], dataMat[:, 2]
thetaTest = matrix([[1], [1]])
J = computeCost(X, y, theta)

theta, J_history = ex1.gradientDescent(x, y, thetaTest, 0.01, 1000)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + (theta[1, 0] * x)

#将散点图和回归函数在同一个图表上展示：
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(x, f, label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

#cost随迭代次数的变化趋势
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(range(1000), J_history, label='Cost')
ax.set_xlabel('Iter_times')
ax.set_ylabel('Cost')

#用scikit-learn进行预测
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

#检验scikit-learn算法表现
f = model.predict(X).flatten()#flatten:把诸如（x，1）的二维数组转为1维
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

thetaN = ex1.normalEqn(X, y)