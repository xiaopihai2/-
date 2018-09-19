from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit        # 交叉验证所需的子集划分方法
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit         # 分层分割
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut,LeavePGroupsOut,GroupShuffleSplit # 分组分割
from sklearn.model_selection import TimeSeriesSplit     #时间序列分割
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import recall_score    #模型度量

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=42)
print("训练集：{}\n测试集：{}".format(X_train.shape, X_test.shape))
model = svm.SVC(kernel='linear', C = 1).fit(X_train, y_train)
print("准确率：", model.score(X_test, y_test))

#对数据归一化处理：
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaler = scaler.transform(X_train)
model = svm.SVC(kernel="linear", C = 1).fit(X_train_scaler, y_train)
X_test_scaler = scaler.transform(X_test)
print("准确率：", model.score(X_test_scaler, y_test))       #数据归一化对模型还是有影响，准确度下降

#========================普通的交叉验证============================
model = svm.SVC(kernel='linear', C = 1)
scores = cross_val_score(model, iris.data, iris.target, cv=5, )
print(scores)
print("准确区间:[{} {}]".format(scores.mean()-scores.std()**2, scores.mean()+scores.std()**2))
#=======================多种度量结果===============================
scoring = ['precision_macro', 'recall_macro'] #precision_macro为精度，recall_macro为召回率
scores = cross_validate(model, iris.data, iris.target, cv =5, scoring=scoring, return_train_score=True)
sorted(scores.keys())
print("测试结果：", scores)  #scores为字典，包含训练得分， 拟合次数， 得分次数


#=====================k折交叉验证、留一交叉验证、留P交叉验证、随机排列交叉验证=====================
#k折划分子集,将数据集分成k份,一份做测试集，其余做训练集
kf = KFold(n_splits=10)
for train, test in kf.split(iris.data):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    break

#留一划分子集,将数据中留一个数据样本作为测试数据吗其余为训练集
one = LeaveOneOut()
for train, test in one.split(iris.data):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    break

#留P:相当于留下P条数据样本作为测试集
P = LeavePOut(p=2)
for train,test in P.split(iris.data):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    break

#随机排列：就是先将数据集打乱
shuffle = ShuffleSplit(n_splits=3, test_size=0.25, random_state=22)
for train,test in shuffle.split(iris.data):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    break

# ==================================分层K折交叉验证、分层随机交叉验证==========================================
##各个类别的比例大致和完整数据集中相同
startkf = StratifiedKFold(n_splits=3)
for train,test in startkf.split(iris.data, iris.target):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    break

startshuffle = StratifiedShuffleSplit(n_splits=3)
for train,test in startshuffle.split(iris.data, iris.target):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    break
# ==================================组 k-fold交叉验证、留一组交叉验证、留 P 组交叉验证、Group Shuffle Split==========================================
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10, 25, 23]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d", 'd', 'd']
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3,3,3]
gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y,groups=groups):
    print("train_size:{}\ntest_size:{}".format(train.shape, test.shape))
    print(train, test)
# 留一分组
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("留一组分割：%s %s" % (train, test))

# 留p分组
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("留 P 组分割：%s %s" % (train, test))

# 随机分组
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("随机分割：%s %s" % (train, test))


# ==================================时间序列分割==========================================
tscv = TimeSeriesSplit(n_splits=3)
TimeSeriesSplit(max_train_size=None, n_splits=3)
for train, test in tscv.split(iris.data):
    print("时间序列分割：%s %s" % (train, test))
