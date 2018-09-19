#决策树

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# 下面的数据分为为每个用户的来源网站、位置、是否阅读FAQ、浏览网页数目、选择的服务类型（目标结果）
attr_arr=[['slashdot','USA','yes',18,'None'],
         ['google','France','yes',23,'Premium'],
         ['digg','USA','yes',24,'Basic'],
         ['kiwitobes','France','yes',23,'Basic'],
         ['google','UK','no',21,'Premium'],
         ['(direct)','New Zealand','no',12,'None'],
         ['(direct)','UK','no',21,'Basic'],
         ['google','USA','no',24,'Premium'],
         ['slashdot','France','yes',19,'None'],
         ['digg','USA','no',18,'None'],
         ['google','UK','no',18,'None'],
         ['kiwitobes','UK','no',19,'None'],
         ['digg','New Zealand','yes',12,'Basic'],
         ['slashdot','UK','no',21,'None'],
         ['google','UK','yes',18,'Basic'],
         ['kiwitobes','France','yes',19,'Basic']]

dataMat = np.mat(attr_arr)
arrMat = dataMat[:, :4]
resultMat = dataMat[:, 4]

#构造pandas结构
attrs_names = ['src', 'address', 'FAQ', 'num']
attr_pd = pd.DataFrame(data = arrMat, columns=attrs_names)
print(attr_pd)

#将数据集中的字符串转换成数字，sklearn中决策树只认数字
le = LabelEncoder()         #简单来说 LabelEncoder 是对不连续的数字或者文本进行编号
for col in attr_pd.columns:
    attr_pd[col] = le.fit_transform(attr_pd[col])
print(attr_pd)

#构建决策树
clf = tree.DecisionTreeClassifier()
clf.fit(attr_pd, resultMat)
print(clf)

#使用决策树进行预测
result = clf.predict([[1,1,1,0]])
print(result)

#将决策树保存成图片
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
target_name = ['None', "Basic", 'Premium']
tree.export_graphviz(clf, out_file=dot_data,feature_names=attrs_names,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')