import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

#===========================Bagging=====================
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
"""
    KNeighborsClassifier：弱分类器模型
    max_samples;数据的部分抽取比例(50%抽取)
    max_feature:数据的部分特征抽取比例(50%抽取)
    bootstrap 和 bootstrap_features 控制着样例和特征的抽取是有放回还是无放回的
    当使用样本子集时，通过设置 oob_score=True ，可以使用袋外(out-of-bag)样本来评估泛化精度。
"""
Bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
score = cross_val_score(Bagging, X, y)#这里应该是准确率
print("平均准确率：", score.mean())

#=======================随机森林==========================
#sklearn.ensemble 模块包含两个基于 随机决策树 的平均算法： RandomForest 算法和 Extra-Trees 算法。
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#决策树：
#max_depth:树的最大深度；min_samples_split:分裂所需最小样本数
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
score = cross_val_score(clf, X, y)
print("平均准确率：", score.mean())

#RandomForestClassifier：
#n_estimators:10棵树， max_features;每棵树分割节点时考虑的特征的随机子集的大小
#分类问题使用 max_features = sqrt（n_features （其中 n_features 是特征的个数）是比较好的默认值。
RT = RandomForestClassifier(n_estimators=10, max_features=2)
RT.fit(X, y)
score = cross_val_score(RT, X, y)
print("平均准确率：", score.mean())

#ExtraTreesClassifier:

ET = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
score = cross_val_score(ET, X, y)
ET.fit(X,y)
print("平均准确率：", score.mean())
print('模型中各属性的重要程度：',ET.feature_importances_)

#=========================AdaBoos===========================
"""
弱学习器的数量由参数 n_estimators 来控制。 learning_rate
参数用来控制每个弱学习器对最终的结果的贡献程度（校对者注：
其实应该就是控制每个弱学习器的权重修改速率，这里不太记得了，不确定）。
弱学习器默认使用决策树。不同的弱学习器可以通过参数 base_estimator
来指定。获取一个好的预测结果主要需要调整的参数是 n_estimators 和
base_estimator 的复杂度 (例如:对于弱学习器为决策树的情况，树的深度
max_depth 或叶子节点的最小样本数 min_samples_leaf 等都是控制树的
复杂度的参数)
"""
from sklearn.ensemble import AdaBoostClassifier
AdaBoost = AdaBoostClassifier(n_estimators=100)
score = cross_val_score(AdaBoost, X, y)
print("平均准确率：", score.mean())

#====================================GBDT========================
"""
   max_features:划分时考虑的最大特征数,默认为None, 可取值;log2；表示取log2N(2为底,N个特征)
                 sqrt, auto:根号下N; int型：就是取几个; float型;取百分比
   max_depth：决策树的最大深度
   min_samples_split： 内部节点再划分所需最小样本数
   min_samples_leaf:叶子节点最少样本数
   min_weight_fraction_leaf：叶子节点最小的样本权重和（这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了）
   max_leaf_nodes:最大叶子节点数，防止过拟合
   min_impurity_split: 节点划分最小不纯度，这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。
"""
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
#这里学习步长(效率)取值敏感
GBDT = GradientBoostingClassifier(n_estimators=1000,learning_rate=1.0, max_depth=1, random_state=0)
score = cross_val_score(GBDT, X, y)
print("平均准确率：", score.mean())

#回归：
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
X_train,X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)
#如果你指定 max_depth=h ，那么将会产生一个深度为 h 的完全二叉树。
GBDT = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, loss='ls',min_samples_split=2, max_depth=4)
GBDT.fit(X_train, y_train)
print("GBDT的MSE:",mean_squared_error(y_test, GBDT.predict(X_test)))
print("记录每次训练的得分：", GBDT.train_score_)
print("各个特征的重要性：", GBDT.feature_importances_)
plt.plot(np.arange(500), GBDT.train_score_, 'b-')
plt.show()

#======================投票器Voting Classifier===========================
data = load_iris()
X, y = data.data, data.target

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression(random_state=1)
model2 = GaussianNB()
model3 = RandomForestClassifier(random_state=1)
#eclf1：无权重投票和有权重投票
models = VotingClassifier(estimators=[('lr', model1), ('gbn', model2), ('rf', model3)],voting='hard')
models = VotingClassifier(estimators=[('lr', model1), ('gbn', model2), ('rf', model3)],voting='soft', weights=[2,1,2])
for model, label in zip([model1, model2, model3, models], ['LogisticRegression', 'GaussianNB','RandomForestClassifier','VotingClassifier']):
    scores = cross_val_score(model, X, y, cv = 5, scoring='accuracy')
    print("准确率: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
from sklearn.model_selection import GridSearchCV

#网格搜索
params = {"lr__C":[1.0, 100.0], "rf__n_estimators":[20, 200]} #搜索寻找最优的lr模型中的C参数和rf模型中的n_estimators
grid = GridSearchCV(estimator=models, param_grid=params, cv=5).fit(X, y)

print("最优的参数：", grid.best_params_)