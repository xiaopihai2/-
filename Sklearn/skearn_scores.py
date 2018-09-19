from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import  load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm


#=====================分类度量========================
data = load_iris()
X, y = data.data, data.target

model1 = svm.SVC(probability=True, random_state=0)
scores = cross_val_score(model1, X, y, cv = 5)
print(scores)

# 自定义度量函数，输入为真实值和预测值
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

loss = make_scorer(my_custom_loss_func, greater_is_better = False)      #越小越好，相当于损失值，与得分值相比取反
score = make_scorer(my_custom_loss_func, greater_is_better = True)      #模型得分，越大越好

model2 = svm.SVC()
model2.fit(X, y)
print(loss(model2, X, y))
print(score(model2, X, y))

# ============================多种度量值=========================
scoring = ['precision_macro', 'recall_macro'] # precision_macro为精度，recall_macro为召回率
scores = cross_validate(model2, X, y,scoring=scoring,cv=5, return_train_score=True)
sorted(scores.keys())
print('多种度量的测试结果：',scores)

#========================分类指标========================
y_predict = model2.predict(X)
print("准确率指标：", metrics.accuracy_score(y, y_predict))
print("Kappa指标：", metrics.cohen_kappa_score(y, y_predict))
print("混淆矩阵：", metrics.confusion_matrix(y, y_predict))

target_names = ['class1', 'class2', 'class3']
print("分类报告：", metrics.classification_report(y, y_predict, target_names=target_names))
print("汉明损失：", metrics.hamming_loss(y, y_predict))
print("Jaccard:" , metrics.jaccard_similarity_score(y, y_predict))
# 下面的系数在在二分类中不需要使用average参数，在多分类中需要使用average参数进行多个二分类的平均
# average可取值：macro（宏）、weighted（加权）、micro（微）、samples（样本）、None（返回每个类的分数）

print('精度计算：',metrics.precision_score(y, y_predict, average='macro'))
print('召回率：',metrics.recall_score(y, y_predict,average='micro'))
print('F1值：',metrics.f1_score(y, y_predict,average='weighted'))

print('FB值：',metrics.fbeta_score(y, y_predict,average='macro', beta=0.5))
print('FB值：',metrics.fbeta_score(y, y_predict,average='macro', beta=1))
print('FB值：',metrics.fbeta_score(y, y_predict,average='macro', beta=2))
print('精确召回曲线：',metrics.precision_recall_fscore_support(y, y_predict,beta=0.5,average=None))
print('零一损失：',metrics.zero_one_loss(y, y_predict))

# ROC曲线(二分类)
y1 = np.array([0, 0, 1, 1])  # 样本类标号
y_scores = np.array([0.1, 0.4, 0.35, 0.8]) # 样本的得分（属于正样本的概率估计、或置信度值）
fpr, tpr, thresholds = metrics.roc_curve(y1, y_scores, pos_label=1)
print('假正率：',fpr)
print('真正率：',tpr)
print('门限：',thresholds)
print('AUC值：',metrics.roc_auc_score(y1, y_scores))


labels = np.array([0, 1, 2])  # 三种分类的类标号
pred_decision = model2.decision_function(X)  # 计算样本属于每种分类的得分，所以pred_decision是一个3列的矩阵
print('hinge_loss：',metrics.hinge_loss(y, pred_decision, labels = labels))

# 逻辑回归损失，对真实分类和预测分类概率进行对比的损失
y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
print('log_loss：',metrics.log_loss(y_true, y_pred))


# ===============================回归度量==============================
print(' ===============================回归度量==============================')
diabetes = datasets.load_diabetes()  # 加载糖尿病数据集；用于回归问题
X, y = diabetes.data, diabetes.target  # 442个样本，10个属性，数值输出

model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X, y)   # 线性回归建模
predicted_y = model.predict(X)  # 使用模型预测

print('解释方差得分：',metrics.explained_variance_score(y, predicted_y))  # 解释方差得分
print('平均绝对误差：',metrics.mean_absolute_error(y, predicted_y))  # 平均绝对误差
print('均方误差：',metrics.mean_squared_error(y, predicted_y))  # 均方误差
print('均方误差对数：',metrics.mean_squared_log_error(y, predicted_y))  # 均方误差对数
print('中位绝对误差：',metrics.median_absolute_error(y, predicted_y))  # 中位绝对误差
print('可决系数：',metrics.r2_score(y, predicted_y, multioutput='variance_weighted')) #可决系数
print('可决系数：',metrics.r2_score(y, predicted_y, multioutput='raw_values')) #可决系数
print('可决系数：',metrics.r2_score(y, predicted_y, multioutput='uniform_average')) #可决系数
