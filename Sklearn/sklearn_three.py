#特征提取

#==================从字典类型加载特征，形成系数矩阵结构==========
from sklearn.feature_extraction import DictVectorizer       #将dict转换成sklearn使用的numpy/scipy表示形式
measurements = [
    {'name':'student1', 'age':12},
    {'boy':True, 'parents':'baba'},
    {'size':16}
]
vec = DictVectorizer().fit(measurements) # 定义一个加载器，后对一个字典对象提取特征。（值为数值型、布尔型的属性为单独的属性。值为字符串型的属性，形成"属性=值"的新属性）
print('提取的特征：',vec.get_feature_names())  # 查看提取的新属性
print('稀疏矩阵形式：\n', vec.transform(measurements))         #从矩阵的形式就好理解该稀疏形式，就是将矩阵中不为0的索引和值拿出来
print("二维矩阵形式：\n", vec.transform(measurements).toarray())

#=====================文本特征提取==============================
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?',]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print('所有特征：', vectorizer.get_feature_names())
print('样本特征向量:\n', X.toarray())
print("document属性的列索引：", vectorizer.vocabulary_.get('document'))

#提取多个单词一起词组，识别is this与this is的区别
bigram_vectorizer = CountVectorizer(ngram_range = (1, 2), token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print("所有分词：", analyze('Bi-grams are cool!'))

#========================TF-IDF加权==============================
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)    #将 “1” 计数添加到 idf 而不是 idf 的分母,即idf+1
counts = [[3, 0, 1],
          [3, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]
tfidf = transformer.fit_transform(counts)
print("稀疏矩阵存储:\n", tfidf)
print("二维矩阵存储:\n", tfidf.toarray())
print("特征权重:\n", transformer.idf_)


#特征选择

#====================去除方差小于阈值的特征====================
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest   #移除那些除了评分最高的 K 个特征之外的所有特征
from sklearn.feature_selection import chi2      #卡方分布

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape, X.var(axis = 0), sep = '\n')

sel = VarianceThreshold(threshold = 0.2)
X_variance = sel.fit_transform(X)
print("去除低方差特征：", X_variance.shape)

#==================排序选择优秀特征===============
print('原样本维度：', X.shape)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print("新样本维度：", X_new.shape)


#======================递归式特征消除===========================
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
print('X:\n',X[0])
y = digits.target

svc = SVC(kernel='linear', C = 1)
ref = RFE(estimator = svc, n_features_to_select = 1, step = 1)
ref.fit(X, y)
ranking = ref.ranking_.reshape(digits.images[0].shape)

print('\n:', ranking)

#绘制像素点排名
plt.matshow(ranking, cmap = plt.cm.Blues)
plt.colorbar()
plt.title("ranking of pixels with RFE")
plt.show()

#=====================使用SelectFromModel选取特征======================
# SelectFromModel 是一个 meta-transformer（元转换器） ，它可以用来处理任何带有 coef_ 或者 feature_importances_ 属性的训练
# 之后的评估器。 如果相关的coef_ 或者 featureimportances 属性值低于预先设置的阈值，这些特征将会被认为不重要并且移除掉。除
# 了指定数值上的阈值之外，还可以通过给定字符串参数来使用内置的启发式方法找到一个合适的阈值。可以使用的启发式方法有 mean 、
# median 以及使用浮点数乘以这些（例如，0.1*mean ）。
import matplotlib.pyplot as plt
from sklearn .datasets import load_boston
from sklearn.feature_selection import  SelectFromModel
from sklearn.linear_model import LassoCV

boston = load_boston()
X = boston.data
y = boston.target
n_features = [13]        #记录循环特征的个数，最初是全集即13个特征开始
thresholds = [0]        #记录阈值，最开始是0

clf = LassoCV()
sfm = SelectFromModel(clf, threshold = 0.1)
sfm.fit(X, y)
X_transform = sfm.transform(X)
n_feature = X_transform.shape[1]
n_features.append(n_feature)
thresholds.append(0.1)
while n_feature> 2:
    sfm.threshold += 0.1
    sfm.fit(X, y)
    X_transform = sfm.transform(X)
    n_feature = X_transform.shape[1]
    n_features.append(n_feature)
    thresholds.append(sfm.threshold)

plt.title("Features with thresholds %0.3f" % sfm.threshold)
plt.plot(thresholds, n_features, 'r')
plt.xlabel("thresholds")
plt.ylabel("n_features")
plt.show()


#=========================基于L1的特征选取=========================
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

iris = load_iris()
X, y = iris.data, iris.target
print("原始数据维度：", X.shape)
lsvc = LinearSVC(C = 0.01, penalty = "l1", dual = False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print("新的数据维度：", X_new.shape)

#============================基于Tree树的特征选取===================
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

iris = load_iris()
X, y = iris.data, iris.target
print("原始数据维度：", X.shape)
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
print("属性重要程度：",clf.feature_importances_)

model = SelectFromModel(clf, prefit=True)
X_transform = model.transform(X)
print("新的数据维度：", X_transform.shape)
