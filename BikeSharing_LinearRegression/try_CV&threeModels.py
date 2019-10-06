#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/10/6


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score

def try_Model(*data):
    def printScore(s,regr):
        print('------'+s+'------')
        print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
        print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test).reshape(-1,1) - y_test) ** 2))
        print('Score: %.2f' % regr.score(X_test, y_test))
    X_train,X_test,y_train,y_test=data

    global score
    res = []

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    res.append(regr.score(X_test, y_test))
    # printScore('LinearRegression',regr)

    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    res.append(regr.score(X_test, y_test))
    # printScore('Ridge',regr)

    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    res.append(regr.score(X_test, y_test))
    # printScore('Lasso',regr)

    score.append(res)

    ## 异常值处理
    # error = regr.predict(X_train).reshape(-1,1)-y_train
    # error.abs().to_csv('error2.csv', header = 0)  # 不保存列名

data = pd.read_csv("processed_day.csv")
# print(data)
y = data[['cnt']]
X = data.drop(['cnt'], axis=1)
# print(type(X))
y = np.array(y)
X = np.array(X)
# print(type(X))
# print(X.shape,np.array(X).shape)
# print('x',X)
# print('y',y)
# print(X)

score = []
folder=KFold(n_splits=10,shuffle=False) # 切分之前不混洗数据集
for train_index,test_index in folder.split(X):
      # print("Train Index:",train_index)
      # print("Test Index:",test_index)
      # print("X_train:",X[train_index])
      # print("X_test:",X[test_index])
      # print("")
      try_Model(X[train_index],X[test_index],y[train_index],y[test_index])

print(score)
x = np.linspace(1,10,10)
score = np.array(score)
y1,y2,y3 = score[:,0],score[:,1],score[:,2]
plt.plot(x, y1, 'ro', label= 'LinearRegression')
plt.plot(x, y2, 'bo', label= 'Ridge')
plt.plot(x, y3, 'yo', label= 'Lasso')
plt.legend()
plt.title('cross_val_score for three models while using KFold')
plt.show()