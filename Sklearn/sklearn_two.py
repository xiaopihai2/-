

#数据处理在sklearn下，当然你也可以在pandas下处理后在转换成numpy！！！！！


#标准化，去均值和方差比例缩放
    #将特征缩放至特定范围
    #缩放稀疏(矩阵)数据
    #缩放有离群点数据
    #核矩阵的中心化

#非线性变换
#归一化
#二值化
    #特征二值化

#标称特征编码
#缺失值插补
#生成多项式特征



from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -1., -2.],
                    [2., 0., 0.],
                    [2., 1., -1.]])
X_test = [[-1., 1., 0.]]


#=======================标准化======================================
# #计算数据集的尺度（也就是数据集的均值和方差）(各列)
# scaler = preprocessing.StandardScaler.fit(X_train)    #计算均值和方差
# print("均值：", scaler.mean_)
# print('方差：', scaler.scale_)
#
# #通过尺度去处理另一个数据集， 当然另一个数据集仍然可以是自己
# X_scaled = scaler.transform(X_train)    #transform会将数据集变成均值为0， 方差为1
# print('均值：', X_scaled.mean(axis = 0))
# print("方差：", X_scaled.std(sxis = 0))

#综合上面两步：缩放样本，是样本均值0， 方差为1(各列)
X_scaled = preprocessing.scale(X_train, axis=0)
print('均值：', X_scaled.mean(axis=0))
print("方差：", X_scaled.std(axis=0))

#==========================特征缩放===============================
#MinMaxScaler最大最小法
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)  #训练同时转换
print(X_train_minmax)
print('每列最大值：', X_train_minmax.max(axis=0)) #最大值为1
print("每列最小值：", X_train_minmax.min(axis = 0))   #最小值为0
#缩放对象是记录了， 平移距离和缩放大小， 在对数据进行操作
print('先平移：', min_max_scaler.min_)
print("再缩放：", min_max_scaler.scale_)


#MaxAbsScaler通过除以每个特征的最大值将数据特征缩放至[-1, 1]
X_train = np.array([[0., -1., 0.],
                    [0., 0., 0.2],
                    [2., 0., 0.]])
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print("每列最大值：", X_train_maxabs.max(axis=0))
print("每列最小值：", X_train_maxabs.min(axis=0))
print("缩放比例：", max_abs_scaler.scale_)
X_test_maxabs = max_abs_scaler.transform(X_test)
print("缩放后的矩阵仍然具有稀疏性：\n", X_train_maxabs)

#==================缩放离群点的值===========================
X_train = np.array([[1., -11., -2.],
                    [2., 2., 0.],
                    [13., 1., -11.]])
robust_scale = preprocessing.RobustScaler()
X_train_robust = robust_scale.fit_transform(X_train)
print("缩放后的数据：", X_train_robust)

#=================非线性转换===========================
X_train = np.array([[1., -1., -2.],
                    [2., 0., 0.],
                    [3., 1., -1]])
quantile_transformer = preprocessing.QuantileTransformer(random_state=0) #将数据默认映射到0-1的均匀分布上
X_train_quantile = quantile_transformer.fit_transform(X_train)
print("原分位数情况：", np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
print("变换后分位数情况：", np.percentile(X_train_quantile[:, 0], [0, 25, 50, 75, 100]))
print(X_train_quantile)

#=====================归一化==================================
X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l1')    #缩放每个样本，使样本数据（每行）的1范或2范为1
print("样本归一化：\n", X_normalized)
#当然仍然可以先通过样本获取转换对象，再用转换对象归一化数据
normalizer = preprocessing.Normalizer().fit(X)   #获取转换对象
normalizer.transform(X) #转换任何数据

#===========================特征二值化========================
binarizer = preprocessing.Binarizer(threshold=1).fit(X)     #默认二值化界点为0：<=0的数取0， 反之取1，这里为1
print(binarizer)
X_binarizer = binarizer.transform(X)
print(X_binarizer)

#===========================ONE-HOT编码======================
from sklearn.preprocessing import  OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 1, 2],
         [1, 0, 0],
         [0, 2, 1],
         [1, 0, 1]])
print('取值范围整数个数：', enc.n_values_)   #这里是判断每列（每个属性）中，不同元素的个数
print("编码后：", enc.transform([[0,1, 1]]).toarray())  #将数据进行one-hot编码
print("特征开始位置的索引", enc.feature_indices_)

#==================缺失值插补================
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2],
         [np.nan, 3],
         [7, 6]])       #好像必须要训练一次
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print("缺失值插值后：\n", imp.transform(X))

#========================生成多项式特征=================
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[0, 1],
              [2, 3],
              [4, 5]])
poly = PolynomialFeatures(2, interaction_only=False) #最大二次方， interaction_only 为True表示只保留交互项
print("多项式特征：\n", poly.fit_transform(X))  #如果数据为（x1, x2）最后变成1, x1, x2, x1^2, x2^2, x1x2