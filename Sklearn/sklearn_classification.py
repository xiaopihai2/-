"""
多类分类：多类分类假设每一个样本有且仅有一个标签：一个水果可以被归类为苹果，也可以 是梨，但不能同时被归类为两类。
多标签分类：给每一个样本分配一系列标签。一个文本可以归类为任意类别，例如可以同时为政治、金融、 教育相关或者不属于以上任何类别。
多输出分类：为每个样本分配一组目标值。这可以认为是预测每一个样本的多个属性，比如说一个具体地点的风的方向和大小。
"""

#多标签分类格式。将多分类转换为二分类的格式，类似于one-hot编码
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
y_new = MultiLabelBinarizer().fit_transform(y)
print('新的输出格式：\n',y_new)

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier  #1对多
from sklearn.multiclass import OneVsOneClassifier   #1对1
from sklearn.multiclass import OutputCodeClassifier #误差校正输出

X,y = load_iris().data, load_iris().target
clf = LinearSVC(random_state=0)
model = OneVsRestClassifier(clf)
model2 = OneVsOneClassifier(clf)        #这里是生成多个二分类分类器，如有n个类， 则排列组合：Cn取2个分类器
model3 = OutputCodeClassifier(clf)      #每一个类被表示为二进制 码（一个 由0 和 1 组成的数组）。保存 location （位置）/ 每一个类的编码的矩阵被称为 code book。
model2.fit(X, y)
model.fit(X, y)
model3.fit(X, y)
y_predict = model.predict(X)
y2_predict = model2.predict(X)
y3_predict = model3.predict(X)
print("预测错的比例1:", (y_predict !=y).sum() / len(y))
print("预测错的比例2:", (y2_predict != y).sum() / len(y))
print("预测错的比例3:", (y3_predict != y).sum() / len(y))

#====================多输出回归=============================
from sklearn.datasets import make_regression, make_classification
from sklearn.multioutput import MultiOutputRegressor       #多输出回归
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

X, y = make_regression(n_samples=10, n_features=100, n_targets=3, random_state=0)
model = GradientBoostingRegressor(random_state=1)
model = MultiOutputRegressor(model)
model.fit(X, y)
print("误差平方和:", metrics.mean_squared_error(y, model.predict(X)))
#===================多输出分类================================
from sklearn.multioutput import  MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import  shuffle  #打乱数据
import numpy as np
#10条数据，100个特征， 30个有效特征，3种分类
X, y = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=3)
y2 = shuffle(y, random_state = 1)
y3 = shuffle(y, random_state = 2)
Y = np.vstack((y, y2, y3)).T
model = RandomForestRegressor(n_estimators=100, random_state=1)
model = MultiOutputClassifier(model)
y_predict = model.fit(X, Y).predict(X)
print('多输出多分类器预测输出分类:\n',y_predict)


