from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('svm_data.txt')
X_data = data[:, 0:2]
y = np.sign(data[:, 2]) #用sign函数划分类别

test_data = [[3,-1], [1,1], [7,-3], [9,0]] # 测试特征空间
test_target = [-1, -1, 1, 1]  # 测试集类标号
plt.scatter(X_data[:, 0], X_data[:, 1], c = y)

#创建模型：
model = svm.SVC()
model.fit(X_data, y, sample_weight=None)
test_predict = model.predict(test_data)
print(test_predict)
plt.show()
#获取支持向量：
print("支持向量：", model.support_vectors_)
#获取支持向量索引：
print("支持向量索引:", model.support_)
#为每一个类别获得支持向量的个数：
print("支持向量数量：", model.n_support_)

#=========================Liner SVM===============
Liner_svm = svm.LinearSVC()
Liner_svm.fit(X_data, y)
result = Liner_svm.predict(test_data)
print(result)
#=======================Liner NuSVC===============
Liner_NuSVC = svm.NuSVC()
Liner_NuSVC.fit(X_data, y)
result = Liner_NuSVC.predict(test_data)
print(result)
#==============================样本不均衡，多分类SVM==================
rng = np.random.RandomState(42)
n_samples_1 = 1000
n_samples_2 = 100
n_samples_3 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2), 0.5* rng.randn(n_samples_2, 2)+[2, 2], 0.5*rng.randn(n_samples_3, 2)+[-3, 3]]    #三类样本点中心为(1.5,1.5)、(2,2)、(-3,3)
y = [0] * (n_samples_1) + [1] * (n_samples_2) + [2] * (n_samples_3) #三类别的个数0:1000， 1:100， 2:100
model = svm.SVC(decision_function_shape='ovo', kernel='linear', C=1.0)
model.fit(X, y)
dec = model.decision_function([[1.5, 1.5]])     # decision_function()的功能：计算样本点到分割超平面的函数距离。 包含几个2分类器，就有几个函数距离
print("分类器的个数：", dec.shape[1])
print(dec)
#绘制， 第一个二分类器的分割超平面
w = model.coef_[0]
print("W: \n",w)
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a*xx - model.intercept_[0] / w[1]

#使用权重分割平面：
weight_model = svm.SVC(class_weight={1:10}, kernel='linear')
weight_model.fit(X, y)

#绘制分割超平面：
ww = weight_model.coef_[0]
wa = -ww[0] / ww[1]
print(weight_model.intercept_[0])
wyy = wa * xx - weight_model.intercept_[0] / ww[1]

#绘制第一个二分类器：
plt.scatter(X[:, 0], X[:, 1], c = y)
print(xx)
plt.plot(xx, yy, 'k-', label = 'no Weights')
plt.plot(xx, wyy, 'k--', label = 'with Weights')

plt.legend()
plt.show()