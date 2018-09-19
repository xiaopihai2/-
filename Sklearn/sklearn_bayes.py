"""
sklearn中含有三种朴素贝叶斯模型：
    先验为高斯分布：GaussianNB
    先验为多项式分布：MultinomialNB
    先验为伯努力分布：BernoulliNB

    多项式分布多用于文本分析：
    先验概率P(c)= 类c下单词总数/整个训练样本的单词总数
    类条件概率P(tk|c)=(类c下单词tk在各个文档中出现过的次数之和+1)/(类c下单词总数+|V|)：拉普拉斯平滑
    V是训练样本的单词表（即抽取单词，单词出现多次，只算一个），|V|则表示训练样本包含多少种单词。

    伯努力分布：
    P(c)= 类c下文件总数/整个训练样本的文件总数
    P(tk|c)=(类c下包含单词tk的文件数+1)/(类c下单词总数+2)
"""

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np

data = load_iris()
X_data = data.data
y = data.target

#高斯
GB = GaussianNB()
GB.fit(X_data, y)
y_predict = GB.predict(X_data)
miss_num = (y != y_predict).sum()
print("数据个数：{}, 分错得样本个数：{}".format(X_data.shape[0], miss_num))

#多项式分布：
from sklearn.naive_bayes import MultinomialNB
#参数alpha默认下为1.0为拉普拉斯平滑，fit_prior：布尔型可选参数，默认为True。布尔参数fit_prior表示是否要考虑先验概率
"""
    fit_prior   class_prior         最终先验概率
    False       填或不填没有意义        P(Y = Ck) = 1 / k
    True        不填                  P(Y = Ck) = mk / m
    True        填                   P(Y = Ck) = class_prior
"""
MB = MultinomialNB()
MB.fit(X_data, y)
y_predict = MB.predict(X_data)
miss_num = (y != y_predict).sum()
print("数据个数：{}, 分错得样本个数：{}".format(X_data.shape[0], miss_num))

#伯努力分布：
from sklearn.naive_bayes import BernoulliNB
BB = BernoulliNB()
BB.fit(X_data, y)
y_predict = BB.predict(X_data)
miss_num = (y != y_predict).sum()
print("数据个数：{}, 分错得样本个数：{}".format(X_data.shape[0], miss_num))

"""
    这个指南的目的是在一个实际任务上探索scikit-learn的主要工具，在二十个不同的主题上分析一个文本集合。
    在这一节中，可以看到：
        1、加载文本文件和类别
        2、适合机器学习的特征向量提取
        3、训练线性模型进行分类
        4、使用网格搜索策略，找到一个很好的配置的特征提取组件和分类器
"""
"""
    1、Loading the 20 newsgroups dataset 加载20个新闻组数据集
    为了获得更快的执行时间为第一个例子，我们将工作在部分数据集只有4个类别的数据集中：
"""
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', shuffle=True,categories=categories,random_state=42)
print(twenty_train.target)
print(twenty_train.target_names)
print(len(twenty_train))
print(len(twenty_train.filenames))#训练文件名长度
print("-------------------------")
print('\n'.join(twenty_train.data[0].split("\n")[:3]))
print("-------------------------")
print(twenty_train.target_names[twenty_train.target[0]])
print("-------------------------")
print("前十数据类别：", twenty_train.target[:10])
for i in twenty_train.target[:10]:
    print("前十所属于的类的名称：", twenty_train.target_names[i])
print("------------------")
"""
    如果我们运用词袋模型，则数据矩阵太大，计算机吃不消，因为词袋模型中的数据矩阵为高稀疏矩阵，所以
    我们就只记录矩阵非零数据，而scipy.sparce可以存储，sklearn内置这种数据结构
"""
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit(twenty_train.data)
print(X_train_counts.shape)
print("------------------------------")
"""
    CountVectorizer支持计算单词或序列的N-grams，一旦合适，这个向量化就可以建立特征词典。
    在整个训练预料中，词汇中的词汇索引值与其频率有关。
"""
print(count_vect.vocabulary_.get(u'algorithm'))     #不理解
print("-----------------------------")
from sklearn.feature_extraction.text import TfidfTransformer
tf_transform  = TfidfTransformer(use_idf=False).fit(X_train_counts)     #这里use_id = False,就是不乘idf，只有TF
print(tf_transform)
print("---------------------------")
X_train_tf = tf_transform.transform(X_train_counts)
print(X_train_tf)
print("---------------------------")
print(X_train_tf.shape)
print("----------------------------")
TF_IDF = TfidfTransformer()
X_train_idf = TF_IDF.fit_transform(X_train_counts)
print(X_train_idf)
print(X_train_idf.shape)
print("---------------------------")
"""
    Training a classifier 训练一个分类器
    既然已经有了特征，就可以训练分类器来试图预测一个帖子的类别，先使用贝叶斯分类器，贝叶斯分类器提供了一个良好的基线来完成这个任务。
    scikit-learn中包括这个分类器的许多变量，最适合进行单词计数的是多项式变量。
"""
from sklearn.naive_bayes import MultinomialNB
MB = MultinomialNB()
X_clf = MB.fit(X_train_idf, twenty_train.target)
print(X_clf)
print("---------------------")

#预测文本类别：
docs_new = ['God is love', 'OpenGL on the GPU is fast']
doc_new_counts = count_vect.transform(docs_new)         #还是对样本词频计算
doc_new = TF_IDF.transform(doc_new_counts)              #tfidf向量构成
predict = MB.predict(doc_new)                           #预测
for doc , categorie in zip(docs_new, predict):
    print("{}属于{}类".format(doc, twenty_train.target_names[twenty_train.target[categorie]]))
print("---------------")

#引入管道机制, 相当于省去一半多的步骤
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('counts', CountVectorizer()), ("TF_IDF", TfidfTransformer()), ("clf", MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target) #训练数据
print(text_clf) ##构造的分类器
predicted = text_clf.predict(doc_new)       #预测数据
print(predicted)
print("------------------------------")
"""
    分析总结：
        1、加载数据集，主要是加载训练集，用于对数据进行训练
        2、文本特征提取：
                对文本进行计数统计 CountVectorizer
                词频统计  TfidfTransformer  （先计算tf,再计算tfidf）
        3、训练分类器：
                贝叶斯多项式训练器 MultinomialNB
        4、预测文档：
                通过构造的训练器进行构造分类器，来进行文档的预测
        5、最简单的方式：
                通过使用pipeline管道形式，来讲上述所有功能通过管道来一步实现，更加简单的就可以进行预测
"""
"""
    Evaluation of the performance on the test set 测试集性能评价
    评估模型的预测精度同样容易：
"""
#对数据集的测试集做判断，看评估精度
twenty_test = fetch_20newsgroups(subset='test', shuffle=True,random_state=42, categories=categories)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print((predicted == twenty_test.target).sum()/len(predicted))
print("------------------------")

"""
    精度已经实现了83.4%，那么使用支持向量机(SVM)是否能够做的更好呢，支持向量机(SVM)被广泛认为是最好的文本分类算法之一。
    尽管，SVM经常比贝叶斯要慢一些。
    我们可以改变学习方式，使用管道来实现分类：
"""

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline(
    [('counts', CountVectorizer()),
     ('TF_IDF', TfidfTransformer()),
     ("clf", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]         #loss:损失函数；penalty：正则化项；n_iter:迭代次数；alpha：乘法因子，即正则化前的系数默认为0.0001
)
text_clf = text_clf.fit(twenty_train, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)
print((predicted != twenty_test.target).sum() / len(predicted))
"""
    sklearn进一步提供了结果的更详细的性能分析工具：
"""

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))