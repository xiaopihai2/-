#数据预处理：sklearn-Processing data
#特征选择：sklearn-Feature selection
#降维：sklearn-Dimensionality reduction

from sklearn.datasets import load_iris
#导入IRIS数据集
iris = load_iris()
#特征矩阵
iris_data = iris.data
#目标向量
iris_target = iris.target
"""
##############无量纲量###############
    将服从正态分布的特征值转换成正态分布，标准化需要计算特征的均值和标准差，
    x' = (x-X) / s， 减去均值除以标准差
"""
from sklearn.preprocessing import StandardScaler
#标准化，返回值为标准化后的数据(列向量处理)
StandardScaler().fit_transform(iris.data)

"""
##############区间放缩###############
    最大最小法：x = (x - Min) / (Max - Min)
"""
from sklearn.preprocessing import MinMaxScaler
#区间缩放，返回值缩放到[0, 1]区间数据(列向量处理)
MinMaxScaler().fit_transform(iris.data)
"""
1、在后续的分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA、LDA这些需要用到协方差分析进行降维的时候，
同时数据分布可以近似为正太分布，标准化方法(Z-score standardization)表现更好。
2、在不涉及距离度量、协方差计算、数据不符合正太分布的时候，可以使用区间缩放法或其他归一化方法。比如图像处理中，
将RGB图像转换为灰度图像后将其值限定在[0 255]的范围。
"""
"""
##############归一化#################
    归一化目的在于样本向量在点乘运算或其他核函数计算相似性时，
    拥有统一的标准，也就是说都转化为“单位向量”。
    公式：x' = x / (sqrt(sum(x^2))), x为向量，后面的是它的模长
"""
from sklearn.preprocessing import Normalizer
#归一化，返回值为归一化的数据(对数据的行向量处理)
Normalizer().fit_transform(iris.data)

"""
####################对定量特征二值化#############
    大于阀值的赋值为1， 小于阀值为0
    x' = 1 if x > c(阀值) else 0
"""
from sklearn.preprocessing import Binarizer
#二值化， 阀值设置为3(对列向量处理)
Binarizer(threshold = 3).fit_transform(iris.data)
"""
############对定性特征哑变量################
    多用于无序类型的数据
"""
from sklearn.preprocessing import  OneHotEncoder
#哑变量， a, b, c 转为(1, 0, 0)，(0, 1, 0), (0, 0, 1) (列向量处理)
OneHotEncoder.fit_transform(iris.target.reshape((-1, 1)))
"""
###########缺失值计算############
"""
from numpy import vstack, array, nan
from sklearn.preprocessing import  Imputer
#缺失值计算，返回值为计算缺失值后的数据(列向量处理)
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))

"""
########数据变换########
    常见的数据变换有基于多项式的、基于指数函数的、基于对数函数的。4个特征，度为2的多项式转换公式如下：
    (x1, x2, x3, x4) -------> (1, x1, x2, x3, x4, x1^2, x1*x2, x1*x3, x1*x4, ......x4^2)
"""
from sklearn.preprocessing import  PolynomialFeatures
#多项式变换(行向量处理)
#参数degree为度，默认值为2
PolynomialFeatures().fit_transform(iris.data)

from numpy import log1p
from sklearn.preprocessing import FunctionTransformer

#自定义转换函数对函数的数据变换
#第一个参数为单元函数
FunctionTransformer(log1p).fit_transform(iris.data)
"""
类 	               功能 	                           说明
StandardScaler 	无量纲化 	标准化，基于特征矩阵的列，将特征值转换至服从标准正态分布
MinMaxScaler 	    无量纲化 	区间缩放，基于最大最小值，将特征值转换到[0, 1]区间上
Normalizer 	    归一化 	基于特征矩阵的行，将样本向量转换为“单位向量”
Binarizer 	        二值化 	基于给定阈值，将定量特征按阈值划分
OneHotEncoder 	 哑编码 	将定性数据编码为定量数据
Imputer 	      缺失值计算 	计算缺失值，缺失值可填充为均值等
PolynomialFeatures 	多项式数据转换 	多项式数据转换
FunctionTransformer 	自定义单元数据转换 	使用单变元的函数来转换数据
"""

"""
#####################特征选择###############
"""
    #方差选择
    #使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。
from sklearn.feature_selection import VarianceThreshold
#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold = 3).fit_transform(iris.data)


    #卡方检验
    #检验特征对标签的相关性，选择其中K个与标签最相关的特征。
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#选择k个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k = 2).fit_transform(iris.data, iris.target)


    #递归特征消除法
    #递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator= LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)


    #基于惩罚项的特征选择法
    #使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。使用feature_selection库的
    # SelectFromModel类结合带L1惩罚项的逻辑回归模型，来选择特征的代码如下：
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
#带L1惩罚项的逻辑回归作为基模型的特征选择
SelectFromModel(LogisticRegression(penalty='l1', C=0.1)).fit_transform(iris.data, iris.target)


    #基于树模型的特征选择法
    #树模型中GBDT可用来作为基模型进行特征选择
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
#GBDT作为基模型的特征选择
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

"""
类 	                所属方式 	说明
VarianceThreshold 	Filter 	方差选择法
SelectKBest 	       Filter 	可选关联系数、卡方校验、最大信息系数作为得分计算的方法
RFE 	              Wrapper 	递归地训练基模型，将权值系数较小的特征从特征集合中消除
SelectFromModel 	    Embedded 	训练基模型，选择权值系数较高的特征
"""
#PCA
from sklearn.decomposition import PCA
#主成分分析法，返回降维后的数据
#参数n_components为主成分数目
PCA(n_components=2).fit_transform(iris.data)


#线性判别分析法(LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
LDA(n_components=2).fit_transform(iris.data, iris.target)