from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


data = load_iris()
x = data.data
y = data.target

pca = PCA(n_components=2)   #降到2维
y_pca = pca.fit_transform(x)

ipca = IncrementalPCA( n_components=2, batch_size=10)       #克服PCA处理大型数据的批处理内存问题，IncrementalPCA 对象使用不同的处理形式使之允许部分计算。这一形式几乎和 PCA 以小型批处理方式处理数据的方法完全匹配
y_ipca = ipca.fit_transform(x)

pca_svd = PCA(n_components=2, whiten=True, svd_solver='randomized')     #PCA使用SVD分解
y_pca1 = pca_svd.fit_transform(x)

plt.subplot(131)
plt.scatter(y_pca[:,0], y_pca[:, 1], c =y, alpha=0.8, lw = 2)
plt.title("PCA")
plt.subplot(132)
plt.scatter(y_ipca[:,0], y_ipca[:,1], c=y, alpha=0.8, lw = 2)
plt.title("IncrementalPCA")
plt.subplot(133)
plt.scatter(y_pca1[:,0], y_pca1[:,1], c = y, alpha=0.8, lw = 2) #lw数据点的大小，alpha;可以理解为清晰度，应该是0-1
plt.title("PCA_SVD")
plt.show()

#=======================kernelPCA=========================
#通过使用核方法对非线性降维
#如下面构造的环形数据就是线性不可分
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles   #sklearn生成样本数据得方法，这里是生成环形数据，还有其他的方法如make_blobs
import  numpy as np

X, y = make_circles(n_samples = 400, factor = .3, noise = .05)
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)    #fit_inverse_transform:将降维的数据覆盖在原数据上
x_kpca = kpca.fit_transform(X)
x_bpca = kpca.inverse_transform(x_kpca)      #不知道什么是逆转

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#绘制原数据
plt.subplot(221)
plt.scatter(X[:,0], X[:,1], c = y, alpha=0.8, lw =2)
plt.title("data")

plt.subplot(222)
plt.scatter(X_pca[:,0], X[:,1], c = y, alpha=0.8, lw = 2)
plt.title("PCA")

plt.subplot(223),
plt.scatter(x_kpca[:,0], x_kpca[:,1], c = y, alpha=0.8, lw = 2)
plt.title("kernelPCA")

plt.subplot(224)
plt.scatter(x_bpca[:,0], x_kpca[:,1],c = y,  alpha=0.8, lw = 2)
plt.title("inverse_transform")
plt.show()