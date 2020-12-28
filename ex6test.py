# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:17:31 2019

@author: 34276
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV
from sklearn import svm

#第一部分
data = loadmat('ex6data3.mat')
X, y = data['X'], data['y'].ravel()
Xval, yval = data['Xval'], data['yval'].ravel()
positive = X[np.where(y==1), :][0]
negative = X[np.where(y==0), :][0]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive[:,0], positive[:,1], s=50, marker='x', label='Positive')
ax.scatter(negative[:,0], negative[:,1], s=50, marker='o', label='Negative')
ax.legend()

test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
parameters = {'kernel':('linear', 'rbf'), 'C':test, 'gamma':test}
gsearch1= GridSearchCV(svm.SVC(), parameters, scoring='f1', cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
gsearch1.score(Xval, yval)

#第二部分
spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')
X,y = spam_train['X'],spam_train['y'].ravel()
Xtest,ytest = spam_test['Xtest'],spam_test['ytest'].ravel()
#每个文档已经转换为一个向量，其中1,899个维对应于词汇表中的1,899个单词。 它们的值为二进制，表示文档中是否存在单词。
svc = svm.SVC()
svc.fit(X, y)
print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))