
"""
    网格搜索调参：GridSearchCV
    随机搜索调参：RandomizedSearchCV
    hyperopt调参

"""
from sklearn.model_selection import GridSearchCV

"""
    GridSearchCV(
    estimator,      模型
    param_grid,     参数
    scoring=None,   评价函数
    fit_params=None,
    n_jobs=1,
    iid=True,
    refit=True,
    cv=None,        几折
    verbose=0,
    pre_dispatch='2*n_jobs',
    error_score='raise',
    return_train_score=True)
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = load_iris()
X,y = data.data, data.target
k_range = range(1, 31)  #kNN中的k的取值范围
weights_options= ['uniform', 'distance']    #代估参数的权重， uniform:无权重， distance：距离的倒数作为权重
param_grid = {'n_neighbors':k_range,'weights':weights_options}  # 定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
print(param_grid)

knn = KNeighborsClassifier(n_neighbors=5)
best_model = GridSearchCV(estimator = knn, param_grid=param_grid, cv = 10, scoring='accuracy')
best_model.fit(X, y)

print('网格搜索-度量记录：',best_model.cv_results_)  # 包含每次训练的相关信息
print('网格搜索-最佳度量值:',best_model.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：',best_model.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳模型：',best_model.best_estimator_)  # 获取最佳度量时的分类器模型

#从得到的最优参数构建最优模型
Knn = KNeighborsClassifier(n_neighbors=best_model.best_params_['n_neighbors'], weights=best_model.best_params_['weights'])
Knn.fit(X, y)
print(Knn.predict([[3, 5, 4, 2]]))

#=================================随机搜索==============================
#通过 n_iter 参数指定计算预算, 即取样候选项数
R_model = RandomizedSearchCV(knn, param_grid, cv = 10, scoring='accuracy', n_iter= 10,random_state=1 )
R_model.fit(X, y)
print('网格搜索-度量记录：',R_model.cv_results_)  # 包含每次训练的相关信息
print('网格搜索-最佳度量值:',R_model.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：',R_model.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳模型：',R_model.best_estimator_)  # 获取最佳度量时的分类器模型

Knn = KNeighborsClassifier(n_neighbors=best_model.best_params_['n_neighbors'], weights=best_model.best_params_['weights'])
Knn.fit(X, y)
print(Knn.predict([[3, 5, 4, 2]]))

#=======================自定义度量函数============
from sklearn import metrics
# 自定义度量函数
    #好像不的行
# def scorerfun(estimator, X, y):
#     y_pred = estimator.predict(X)
#     return metrics.accuracy_score(y, y_pred)
#
# rand = RandomizedSearchCV(knn, param_grid, cv=10, scoring='scorerfun', n_iter=10, random_state=5)  #
# rand.fit(X, y)
# print('随机搜索-最佳度量值:',rand.best_score_)  # 获取最佳度量值



