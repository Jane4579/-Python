# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:08:55 2019

@author: 34276
"""
import ex7
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Part1：徒手实现K-means对样本点进行聚类
data = loadmat('ex7data2.mat')
X = data['X']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])

initial_centroids = ex7.kMeansInitCentroids(X, 3)
idx, centroids = ex7.runkMeans(X, initial_centroids, 10)

cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.legend()

#Part2：徒手实现用K-means压缩图像（减少颜色）
from IPython.display import Image
Image(filename='bird_small.png')
image_data = loadmat('bird_small.mat')
A = image_data['A']
'''
This creates a three-dimensional matrix A whose ﬁrst two indices identify a pixel 
position and whose last index represents red, green, or blue. For example, A(50, 33, 3) 
gives the blue intensity of the pixel at row 50 and column 33.
'''
A = A / 255
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))#A为三维矩阵，将其展开为二维
initial_centroids = ex7.kMeansInitCentroids(X, 16)#将很多颜色压缩到16种颜色
idx, centroids = ex7.runkMeans(X, initial_centroids, 10)
X_recovered = centroids[idx.astype(int),:]
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
plt.imshow(X_recovered)#io.imshow(X_recovered)

#Part3: 用scikit-learn实现K-means
from skimage import io
# cast to float, you need to do this otherwise the color would be weird after clustring
pic = io.imread('bird_small.png') / 255#将图片转化为矩阵
io.imshow(pic)
data = pic.reshape(128*128, 3)

from sklearn.cluster import KMeans#导入kmeans库
model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data)
centroids = model.cluster_centers_
C = model.predict(data)#C为label数组，每个元素为0到15之间的整数（共16类）
compressed_pic = centroids[C].reshape((128,128,3))#得到压缩后的图像矩阵
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)

#part4：PCA
data = loadmat('ex7data1.mat')
X = data['X']
U, S, V = ex7.pca(X)#原始数据的协方差矩阵的奇异值分解.U为主成分
Z = ex7.projectData(X, U, 1)#用U来将原始数据投影到一个较低维（1维）空间中
X_recovered = ex7.recoverData(Z, U, 1)#恢复数据
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X_recovered[:,0].tolist(), X_recovered[:,1].tolist())
#用恢复数据绘制散点图，会发现点都在对角线上。原因是，第一主成分的投影轴基本上是数据集中的对角线。
#当我们将数据减少到一个维度时，我们失去了该对角线周围的变化，所以在我们的再现中，一切都沿着该对角线。

#part 5: 将PCA应用于脸部图像压缩
faces = loadmat('ex7faces.mat')
X = faces['X']#1024个特征，32×32pixel
plt.imshow(X[0, :].reshape(32,32), cmap='gray')

U, S, V = ex7.pca(X)
Z = ex7.projectData(X, U, 100)
X_recovered = ex7.recoverData(Z, U, 100)
face = np.reshape(X_recovered[0,:], (32, 32))
plt.imshow(face, cmap='gray')