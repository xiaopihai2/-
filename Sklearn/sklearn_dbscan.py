"""
DBSCSAN聚类，密度聚类，有点像kNN,
可以聚类为任意形状，也可以检测异常点，数据集不凸可以用
数据集不是稠密的，不推荐用，了解它的核对象和密度可达
"""

import  numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

data=[
    [-2.68420713,1.469732895],[-2.71539062,-0.763005825],[-2.88981954,-0.618055245],[-2.7464372,-1.40005944],[-2.72859298,1.50266052],
    [-2.27989736,3.365022195],[-2.82089068,-0.369470295],[-2.62648199,0.766824075],[-2.88795857,-2.568591135],[-2.67384469,-0.48011265],
    [-2.50652679,2.933707545],[-2.61314272,0.096842835],[-2.78743398,-1.024830855],[-3.22520045,-2.264759595],[-2.64354322,5.33787705],
    [-2.38386932,6.05139453],[-2.6225262,3.681403515],[-2.64832273,1.436115015],[-2.19907796,3.956598405],[-2.58734619,2.34213138],
    [1.28479459,3.084476355],[0.93241075,1.436391405],[1.46406132,2.268854235],[0.18096721,-3.71521773],[1.08713449,0.339256755],
    [0.64043675,-1.87795566],[1.09522371,1.277510445],[-0.75146714,-4.504983795],[1.04329778,1.030306095],[-0.01019007,-3.242586915],
    [-0.5110862,-5.681213775],[0.51109806,-0.460278495],[0.26233576,-2.46551985],[0.98404455,-0.55962189],[-0.174864,-1.133170065],
    [0.92757294,2.107062945],[0.65959279,-1.583893305],[0.23454059,-1.493648235],[0.94236171,-2.43820017],[0.0432464,-2.616702525],
    [4.53172698,-0.05329008],[3.41407223,-2.58716277],[4.61648461,1.538708805],[3.97081495,-0.815065605],[4.34975798,-0.188471475],
    [5.39687992,2.462256225],[2.51938325,-5.361082605],[4.9320051,1.585696545],[4.31967279,-1.104966765],[4.91813423,3.511712835],
    [3.66193495,1.0891728],[3.80234045,-0.972695745],[4.16537886,0.96876126],[3.34459422,-3.493869435],[3.5852673,-2.426881725],
    [3.90474358,0.534685455],[3.94924878,0.18328617],[5.48876538,5.27195043],[5.79468686,1.139695065],[3.29832982,-3.42456273]
]
X = np.array(data)
db = skc.DBSCAN(eps = 1.5, min_samples=3).fit(X)#聚类为1.5，min_samples为稠密度，即一个和对象中的minSamples>=5
labels = db.labels_ #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
print("每个样本的簇标号")
print(labels)
raito = len(labels[labels[:] == -1]) / len(labels)      #异常点比例
print("噪声比：{}".format(raito, '.2%'))
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("分为{}类".format(n_clusters))
print("轮廓系数：{}".format(metrics.silhouette_score(X, labels)))        #轮廓系数评价聚类的好坏
for i in range(n_clusters):
    X_new = X[labels[:] == i]
    print(X_new)
    plt.plot(X_new[:, 0], X_new[:, 1], 'o')

plt.show()

#=========================================================================
#第二个案例
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler



#产生数据
center = [[1,1], [-1,-1], [1, -1]]
X, label_true = make_blobs(n_samples=750, n_features=2, centers=center, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.3, min_samples=10).fit(X)
mm = np.zeros_like(labels.labels_, dtype=bool)
mm[labels.core_sample_indices_]  = True             #将核点设置为True
labels = labels.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("估计的聚类的个数：{}".format(n_clusters))
print("同质性：{}".format(metrics.homogeneity_score(mm, labels)))   #每个群集只包含单个类的成员
print("完整性：{}".format(metrics.completeness_score(mm, labels)))  #给定类的所有成员都分配给同一个群集。
print("V-measure:{}".format(metrics.v_measure_score(mm, labels)))  #同质性和完整性的调和平均
print("调整兰德指数:{}".format(metrics.adjusted_rand_score(mm, labels)))
print("调整互信息:{}".format(metrics.adjusted_rand_score(mm, labels)))
print("轮廓系数：{}".format(metrics.silhouette_score(X, labels)))


labels_set = set(labels)
cols = [plt.cm.ScalarMappable(each) for each in np.linspace(0, 1, len(labels_set))]
for k ,col in zip(labels_set, cols):
    if k == -1:
        col = [0, 0, 0, 1]
    labels_tu = (labels == k)
    xy = X[labels_tu & mm]        #核点
    plt.plot(xy[:, 0], xy[:, 1], 'o', )    #核点尺寸大些
    xy = X[labels_tu & ~mm]       #非核点
    plt.plot(xy[:, 0], xy[:, 1], 'o', markeredgecolor='k', markersize=6)

plt.show()

#画图出现问题
