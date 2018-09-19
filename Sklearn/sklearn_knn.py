#=================无监督查找最近邻（聚类中使用）=======================
from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
test_x = np.array([[-3.2, -2.1], [-2.6, -1.3], [1.4, 1.0], [3.1, 2.6], [2.5, 1.0], [-1.2, -1.3]])
nbrs = NearestNeighbors(n_neighbors =2, algorithm="ball_tree").fit(X)
distances, indices = nbrs.kneighbors(test_x)
print('邻节点：', indices)
print('邻节点距离：', distances)

#==============使用kd树和Ball树实现==================
from sklearn.neighbors import KDTree, BallTree
kdt = KDTree(X, leaf_size=30, metric = 'euclidean')
distances, indices = kdt.query(test_x, k =2, return_distance=True)
print("邻节点：",indices)
print("邻节点距离：", distances)


#===============最近邻分类==================
    #k邻近分类----k 值的最佳选择是高度依赖数据的：通常较大的 k 是会抑制噪声的影响，但是使得分类界限不明显。
    #RadiusNeighborsClassifier----如果数据是不均匀采样的，那么 RadiusNeighborsClassifier 中的基于半径的近邻分类可能是更好的选择。但当维度较大时, "维灾难"

from sklearn.neighbors import KNeighborsClassifier, KDTree
#4属性， 3类别
data=[
    [ 5.1,  3.5,  1.4,  0.2, 0],
    [ 4.9,  3.0,  1.4,  0.2, 0],
    [ 4.7,  3.2,  1.3,  0.2, 0],
    [ 4.6,  3.1,  1.5,  0.2, 0],
    [ 5.0,  3.6,  1.4,  0.2, 0],
    [ 7.0,  3.2,  4.7,  1.4, 1],
    [ 6.4,  3.2,  4.5,  1.5, 1],
    [ 6.9,  3.1,  4.9,  1.5, 1],
    [ 5.5,  2.3,  4.0,  1.3, 1],
    [ 6.5,  2.8,  4.6,  1.5, 1],
    [ 6.3,  3.3,  6.0,  2.5, 2],
    [ 5.8,  2.7,  5.1,  1.9, 2],
    [ 7.1,  3.0,  5.9,  2.1, 2],
    [ 6.3,  2.9,  5.6,  1.8, 2],
    [ 6.5,  3.0,  5.8,  2.2, 2],
]

dataMat = np.array(data)
X = dataMat[:, :4]
y = dataMat[:, 4]
knn = KNeighborsClassifier(n_neighbors = 2, weights='distance')  #neighbors查找最邻近几个点，默认为5， 权重weights样本权重等于距离的倒数。'uniform'为统一权重
knn.fit(X, y)
result = knn.predict([[3, 2, 2, 5]])
print(result)

#===========================最邻近回归=======================
from sklearn import neighbors
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.sort(5*np.random.rand(40, 1), axis = 0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
print("T：", T)
y = np.sin(X).ravel()       #将多维数组降位一维

#为输出值添加噪声
y[::5] += 1 * (0.5 - np.random.rand(8))

#训练回归模型
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i+1)
    plt.scatter(X, y, c ='k', label = 'data')
    plt.plot(T, y_, c ='g', label = 'prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))
plt.show()
