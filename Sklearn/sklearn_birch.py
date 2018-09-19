"""
特征树，CF Tree聚类
    速度快，只需遍历一遍数据，可以检测异常点
"""

#生成一批样本数据
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#生成1000个样本数据， 维度为2， 4个簇，中心为下面四个点
X, y = make_blobs(n_samples= 1000, n_features=2, centers=[[-1, -1], [0, 0], [1,1], [2, 2]])
plt.scatter(X[:, 0], X[:, 1], marker='o', c = y)
plt.show()
from sklearn.cluster import Birch
#如果不输入k(簇)的个数，聚类效果并不一定好。对于threshold和branching_factor我们前面还没有去调参，使用了默认的threshold值0.5和默认的branching_factor值50.
y_pred = Birch(n_clusters=4).fit_predict(X)
plt.scatter(X[:,0],X[:,1], c = y_pred)
plt.show()
from sklearn import metrics
print("CH指标：", metrics.calinski_harabaz_score(X, y_pred))
#尝试多个threshold和branching_factor
param_grid = {'threshold':[0.5, 0.3, 0.1], 'branching_factor':[50, 20, 10]}
for threshold in param_grid['threshold']:
    for branching_factor in param_grid['branching_factor']:
        y =  Birch(n_clusters=4,threshold=threshold, branching_factor=branching_factor).fit_predict(X)
        plt.scatter(X[:,0], X[:,1], c =y)
        plt.show()
        print("CH指标：", metrics.calinski_harabaz_score(X, y))

#还是默认好-------