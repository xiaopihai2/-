from sklearn.datasets import load_boston    #波士顿房价数据回归使用
from sklearn import linear_model

from sklearn.datasets import load_iris  #花的数据分类
from sklearn import svm

from sklearn.datasets import load_diabetes  #糖尿病数据回归
from sklearn import linear_model

from sklearn.datasets import load_digits    #手写数字识别分类
import matplotlib.pyplot as plt

from sklearn.datasets import load_linnerud  #多元回归
from sklearn import linear_model

from sklearn.datasets import load_sample_image  #sklearn自带图片


boston = load_boston()
data = boston.data
target = boston.target
print(data.shape)
print(target.shape)
print('系数矩阵:\n', linear_model.LinearRegression().fit(data, target).coef_)


iris = load_iris()
data = iris.data
target = iris.target
print(data.shape)
print(target.shape)
print('SVM模型:\n', svm.SVC().fit(data, target))


diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
print(data.shape, target.shape, sep = '\n')
print("系数矩阵\n", linear_model.LinearRegression().fit(data, target).coef_)


digits = load_digits()
data = digits.data
print(data.shape)
plt.matshow(digits.images[3])   #矩阵像素点的样式显示为3
plt.imshow(digits.images[3])  #图片渐变样式显示为3
# plt.gray()                    #灰度图
plt.show()


linnerud = load_linnerud()
data = linnerud.data
target = linnerud.target
print(data.shape, target.shape, sep = '\n')
print("系数矩阵:\n", linear_model.LinearRegression().fit(data, target).coef_)


img = load_sample_image('flower.jpg')
plt.imshow(img)
plt.show()