# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:51:56 2019

@author: 34276
"""

import ex3
import numpy as np
from scipy.io import loadmat

#获取数据
data = loadmat('ex3data1.mat')
X,y = np.matrix(data['X']),np.matrix(data['y'])
X = np.insert(X, 0, 1, axis=1)

#查看有多少类
np.unique(y, axis=0)

#计算参数
all_theta = ex3.oneVsAll(X,y,10,1)

#预测结果并计算准确度
h = ex3.predictOneVsAll(all_theta, X)
ac = ex3.accuracy(h, y)