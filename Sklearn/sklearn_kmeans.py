from sklearn.cluster import Birch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

"""
第1部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：x1
第二列表示球员每分钟得分数：x2
"""

X = [[0.0888, 0.5885],[0.1399, 0.8291],[0.0747, 0.4974],[0.0983, 0.5772],[0.1276, 0.5703],
     [0.1671, 0.5835],[0.1906, 0.5276],[0.1061, 0.5523],[0.2446, 0.4007],[0.1670, 0.4770],
     [0.2485, 0.4313],[0.1227, 0.4909],[0.1240, 0.5668],[0.1461, 0.5113],[0.2315, 0.3788],
     [0.0494, 0.5590],[0.1107, 0.4799],[0.2521, 0.2735],[0.1007, 0.6318],[0.1067, 0.4326],
     [0.1456, 0.8280]
     ]

"""
第2部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""

clf = KMeans(n_clusters = 3)
y_pred = clf.fit_predict(X)
print("k均值模型：", clf)
print("聚类结果：\n", y_pred)

X1 = [n[0]  for n in X]
X2 = [n[1] for n in X]
plt.scatter(X1, X2, c = y_pred, marker='x')
plt.title('k-means聚类')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

#==========================小批量的K-means===========================
    #具体就是每次从数据集中随机选取小批量数据进行聚类，至于聚点更新，自己查看

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs


np.random.seed(0)

batch_size =45
centers = [[1,1], [-1, -1], [1, -1]] #三类聚点
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)  # 生成样本随机数

#K-menas聚类
k_means = KMeans(init="k-means++", n_clusters=3, n_init=10)
begin_time = time.time()
k_means.fit(X)
t_batch = time.time() - begin_time
print("k均值聚类时长：", t_batch)
######################################################
#===========小批量k-means======================
    #batch_size:每次随机多少条数据
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
begin_time = time.time()
mbk.fit(X)
t_mbk_batch = time.time() - begin_time
print("小批量方法的用时：", t_mbk_batch)
###############################################
#结果可视化
fig = plt.figure(figsize=(16, 6))
fig.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.05, top = 0.9)# 窗口四周留白
colors = ['#4EACC5', '#FF9C34', '#4E9A06']  # 三种聚类的颜色

#在两种聚类算法中，样本的所属类标号和聚类中心
k_means_cluster_centers = np.sort(k_means.cluster_centers_,axis=0)  #将聚点排序
mbk_means_clusters_centers = np.sort(mbk.cluster_centers_,axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)  #计算X中的每个点到聚点的距离，距离最短的判别为该类
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_clusters_centers)
order = pairwise_distances_argmin(k_means_cluster_centers,mbk_means_clusters_centers)  # 计算k均值聚类点相对于小批量k均值聚类点的索引。因为要比较两次聚类的结果的区别，所以类标号要对应上

#绘制Kmeans
ax = fig.add_subplot(1,3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor = col, marker = '.')   #绘制当前聚类的数据
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)   #绘制当前类的聚点

ax.set_title("K-Means")
ax.set_xlabel(())
ax.set_ylabel(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
#绘制小批量
ax = fig.add_subplot(1,3,2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_clusters_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor = col, marker = '.')   #绘制当前聚类的数据
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)   #绘制当前类的聚点

ax.set_title("MiniBatchKMeans")
ax.set_xlabel(())
ax.set_ylabel(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_mbk_batch, mbk.inertia_))

#初始化两次结果中
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1,3,3)
for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))# 将两种聚类算法中聚类结果不一样的样本设置为true，聚类结果相同的样本设置为false
identic = np.logical_not(different)# 向量取反，也就是聚类结果相同设置true，聚类结果不相同设置为false
ax.plot(X[identic, 0], X[identic, 1], 'w',markerfacecolor='#bbbbbb', marker='.') # 绘制聚类结果相同的样本点
ax.plot(X[different, 0], X[different, 1], 'w',markerfacecolor='m', marker='.') # 绘制聚类结果不同的样本点
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()
