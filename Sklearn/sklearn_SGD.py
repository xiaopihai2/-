"""
SGD:随机梯度下降：
    优点：高效， 易于实现
    缺点：超参数麻烦，正则化参数和迭代次数参数
          对特征缩放敏感
"""
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
"""
    loss:损失函数；
        hinge:线性支持向量机；
        modified_huber：平滑的 hinge 损失
        log:逻辑回归的损失函数
        企其他所有回归损失函数
    penalty:正则化项
        l2;
        l1;
        elasticnet：(1 - l1_ratio) * L2 + l1_ratio * L1 感觉l1_ratio像一个常数调节l1, l2

"""
model = SGDClassifier(alpha=0.01, loss='hinge', max_iter=200, fit_intercept=True)
model.fit(X, y)
print("回归系数：", model.coef_)
print("偏差：", model.intercept_)

#绘制线，点：
xx1 = np.linspace(-1, 5, 10)
xx2 = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx1, xx2)  #画网格，相当于产生10*10的矩阵
Z = np.empty(X1.shape) #产生像X1一样规格的矩阵

for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    m = np.array([[x1, x2]])
    p = model.decision_function(m)
    Z[i,j] = p[0]
levels = [-1.0, 0.0, 1.0] #将输出分为-1，0，1三个区间
linestyles = ['dashed', 'solid', 'dashed']
plt.contour(X1, X2, Z, levels, color = 'k', linestyles = linestyles) # 绘制等高线图，高度为-1,0,1，也就是支持向量形成的线和最佳分割超平面
plt.scatter(X[:, 0], X[:, 1], c =y, s = 20)
plt.show()