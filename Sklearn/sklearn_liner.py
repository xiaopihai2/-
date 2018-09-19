"""
意义：数据中心化和标准化在回归分析中是取消由于量纲不同、自身变异或者数值相差较大所引起的误差。
原理：数据标准化：是指数值减去均值，再除以标准差；
数据中心化：是指变量减去它的均值。
目的：通过中心化和标准化处理，得到均值为0，标准差为1的服从标准正态分布的数据。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=np.array([
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
])
print(data.shape)
X_data = data[:, 0:1]       #这里不能用data[:, 0]
print(X_data)
y = data[:, 1]
"""
n_jobs 表示调用cpu的个数，-1表示全部cpu,
fit_intercept 是否计算截距
normalize 是否标准化(去均值除以标准差)
copy_X 是否对X复制，如果选择false，则直接对原数据进行覆盖。
"""

LR = LinearRegression(n_jobs=1, fit_intercept=True, normalize=False, copy_X=True)
LR.fit(X_data, y)
print("系数矩阵：", LR.coef_)
print("模型：", LR)
y_predict = LR.predict(X_data)

plt.scatter(X_data, y, marker='X')
plt.plot(X_data, y_predict, c = 'r')

plt.show()

#==================岭回归==========================L2正则化
from sklearn.linear_model import Ridge, RidgeCV
data = np.array([
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
])
X_data = data[:,0:1]
y = data[:, 1]

model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv = 10)
model.fit(X_data, y)
print("系数：", model.coef_)
print("模型：", model)
print("最好的系数：", model.alpha_)
y_predict = model.predict(X_data)

plt.scatter(X_data, y, marker='X')
plt.plot(X_data, y_predict, c = 'r')
plt.show()

#===================Lasso==============L1正则化
"""
lasso 估计解决了加上罚项α||w||1的最小二乘法的最小化
Lasso 类的实现使用了 coordinate descent （坐标下降算法）来拟合系数
对于具有许多线性回归的高维数据集， LassoCV 最常见。 然而，LassoLarsCV 在寻找 α 参数值上更具有优势，
而且如果样本数量与特征数量相比非常小时，通常 LassoLarsCV 比 LassoCV 要快。
"""
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV

La = Lasso(alpha=0.01)
La.fit(X_data, y)
print("系数：", La.coef_)
print("模型：", La)
y_predict = La.predict(X_data)
plt.scatter(X_data, y, marker= 'x')
plt.plot(X_data, y_predict, c = 'r')
plt.show()

#======================MultiTaskLasso 是一个估计多元回归稀疏系数的线性模型================(Y的值是多维)
#使用F范数约束损失函数，利用L1, L2一起正则化系数w，

from  sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.RandomState(42)
n_sample, n_feature, n_tasks = 1000, 30, 40
n_relevant = 5  # 自定义实际有用特征的个数
cofe = np.zeros((n_tasks, n_feature))
times = np.linspace(0, 2*np.pi, n_tasks)
for k in range(n_relevant):
    cofe[:, k] = np.sin((1+rng.randn(1)) * times + 3*rng.randn(1))  #自定义数据矩阵， 用于生成模拟输出值
X = rng.randn(n_sample, n_feature)  # 产生随机输入矩阵
Y = np.dot(X, cofe.T) + rng.randn(n_sample, n_tasks)         # 输入*系数+噪声=模拟输出
print(Y.T)
#=======================使用样本数据训练系数矩阵==================
cofe_lasso = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])          #将Y中的每一维单独拿出来，进行训练
cofe_multi_task_lasso = MultiTaskLasso(alpha=1).fit(X, Y).coef_
print(cofe_lasso, cofe_multi_task_lasso, sep = '\n')
fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.spy(cofe_lasso)
plt.title("Lasso")
plt.xlabel("Feature")
plt.ylabel("Time (or Task)")
plt.text(10, 5, 'Lasso')
plt.subplot(1,2,2)
plt.spy(cofe_multi_task_lasso)
plt.title("MultiTaskLasso")
plt.xlabel("Feature")
plt.ylabel("Time (or Task)")
plt.text(10, 5, 'MultiTaskLasso')
fig.suptitle('Coefficient non-zero location')

feature_to_plot = 0
plt.figure()
lw = 2
plt.plot(cofe[:, feature_to_plot], color='seagreen', linewidth=lw,
         label='Ground truth')
plt.plot(cofe_lasso[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
         label='Lasso')
plt.plot(cofe_multi_task_lasso[:, feature_to_plot], color='gold', linewidth=lw,
         label='MultiTaskLasso')
plt.legend(loc='upper center')
plt.axis('tight')
plt.ylim([-1.1, 1.1])
plt.show()

#===================================逻辑回归
import pandas as pd

# 样本数据集，第一列为x1，第二列为x2，第三列为分类（二种类别）
data=[
    [-0.017612,14.053064,0],
    [-1.395634,4.662541,1],
    [-0.752157,6.538620,0],
    [-1.322371,7.152853,0],
    [0.423363,11.054677,0],
    [0.406704,7.067335,1],
    [0.667394,12.741452,0],
    [-2.460150,6.866805,1],
    [0.569411,9.548755,0],
    [-0.026632,10.427743,0],
    [0.850433,6.920334,1],
    [1.347183,13.175500,0],
    [1.176813,3.167020,1],
    [-1.781871,9.097953,0],
    [-0.566606,5.749003,1],
    [0.931635,1.589505,1],
    [-0.024205,6.151823,1],
    [-0.036453,2.690988,1],
    [-0.196949,0.444165,1],
    [1.014459,5.754399,1]
]

data = np.mat(data)
X_data = data[:,0:2]
y = data[:,2]
b = np.ones((y.shape))
X = np.column_stack((b, X_data)) #常数项
X_data = np.mat(X)
print("X_data:", X_data)

#归一化，去均值化
# from sklearn.preprocessing import StandardScaler
# Stan = StandardScaler()
# X = Stan.fit_transform(X_data)
# print("X:",X)
# X_data = np.mat(X)

#===================逻辑回归=========================
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(X_data, y)
y_predict = model.predict(X_data)
answ = model.predict_proba(X_data)#预测分类概率
print(answ)

h = 0.02
x1_min, x1_max = X_data[:, 1].min() - 0.5, X_data[:,2].max() +0.5
x2_min, x2_max = X_data[:, 1].min() - 0.5, X_data[:, 2].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))        #生成网格

test_data = np.c_[xx1.ravel(), xx2.ravel()]     #形成测试特征数据集
testMat = np.column_stack((np.ones(((test_data.shape[0]),1)),test_data))  #添加第一列为全1代表b偏量
testMat = np.mat(testMat)
y_test = model.predict(testMat)

#绘制网格
Y = y_test.reshape(xx1.shape)
plt.pcolormesh(xx1, xx2, Y, cmap = plt.cm.Paired)

# 绘制散点图 参数：x横轴 y纵轴，颜色代表分类。x图标为样本点，.表示预测点
plt.scatter(X_data[:,1].flatten().A[0], X[:,2].flatten().A[0],c=y.flatten().A[0],marker='x')   # 绘制样本数据集
plt.scatter(X_data[:,1].flatten().A[0], X[:,2].flatten().A[0],c=y_predict.tolist(),marker='.') # 绘制预测数据集

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()
