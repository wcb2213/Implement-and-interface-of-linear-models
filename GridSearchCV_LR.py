#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/10/7


from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def try_GridSearchCV(data):
    '''
    测试 GridSearchCV 的用法。使用 LogisticRegression 作为分类器，主要优化 C、penalty、multi_class 等参数

    :return: None
    '''
    ### 加载数据
    X_train,X_test,y_train,y_test= data
    #### 参数优化 ######
    param_grid = {'penalty': ['l1','l2'],
                  'C': [0.01,0.05,0.1,0.5,1],
                  'solver':['liblinear'],
                  'multi_class': ['ovr']}
    clf=GridSearchCV(LogisticRegression(),param_grid,cv=10)
    grid_result = clf.fit(X_train,y_train)

    print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
    print("Best parameters set found:",clf.best_params_)
    print("Grid scores:")
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))


if __name__=='__main__':
    digits = load_digits()
    try_GridSearchCV(train_test_split(digits.data, digits.target,test_size=0.25,
                random_state=0,stratify=digits.target))