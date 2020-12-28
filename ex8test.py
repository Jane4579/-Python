# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:19:47 2019

@author: 34276
"""
import ex8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

#Part1: Anomaly Detection
data = loadmat('ex8data1.mat')
X, Xval, yval = data['X'], data['Xval'], data['yval']

mu, sigma = ex8.estimateGaussian(X)#假设X服从正态分布，计算该正态分布参数

#计算p值
p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])

#计算验证集的F1值
epsilon, f1 = ex8.selectThreshold(yval, pval)

#寻找测试集中的anomaly（outliers）
outliers = np.where(p < epsilon)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')

#Part2：Collaborative Filtering

data = loadmat('ex8_movies.mat')
Y = data['Y']; R = data['R']
params_data = loadmat('ex8_movieParams.mat')
X = params_data['X']
Theta = params_data['Theta']

#取一小部分数据计算
users = 4
movies = 5
features = 3
X_sub = X[:movies, :features]
Theta_sub = Theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]
params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))
J, grad = ex8.cofiCostFunc(params, Y_sub, R_sub, features, 1.5)

movie_idx = np.array([])
f = open('movie_ids.txt', encoding= 'gbk', errors='ignore')
for line in f:
    tokens = line.split(' ')
    
    tokens[-1] = tokens[-1][:-1]#去除每一行最后的\n
    #print(tokens[1:])
    #movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])
    movie_idx = np.append(movie_idx, ' '.join(tokens[1:]))

#rate movies
ratings = np.zeros((1682, 1))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5
print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))

#add our own ratings vector to the existing data set to include in the model
R = data['R']
Y = data['Y']
Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings != 0, axis=1)

#initiaizlation
movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10
learning_rate = 10.
X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

#standardize
Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))
for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    
#optimize
from scipy.optimize import minimize
fmin = minimize(fun=ex8.cofiCostFunc, x0=params, args=(Ynorm, R, features, learning_rate), 
                method='CG', jac=True, options={'maxiter': 100})
X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

#predict according my ratings
predictions = X * Theta.T 
my_preds = predictions[:, -1] + Ymean
sorted_preds = np.sort(my_preds, axis=0)#从低到高
sorted_preds = sorted_preds[::-1]#从高到低
idx = np.argsort(my_preds, axis=0)[::-1]
movie_idx = np.matrix(movie_idx)
print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))