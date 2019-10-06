#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/22


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def featureExtract1():
    data = pd.read_csv("day.csv")
    data = data[
        ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
         'cnt']]
    # data = data.sample(frac=1)

    ## 剔除异常值
    error_index = [668, 667, 499, 694, 238, 265, 545, 203, 184, 554, 723, 645]
    data.drop(data.index[error_index], inplace=True)

    df = pd.get_dummies(data, columns=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
    df.to_csv('processed_day.csv', index=False)
    label = df[['cnt']]
    df.drop(['cnt'], axis=1, inplace=True)
    # print(df, label)
    # ['temp' 'atemp' 'hum' 'windspeed' 'season_1' 'season_2' 'season_3'
    #  'season_4' 'mnth_1' 'mnth_2' 'mnth_3' 'mnth_4' 'mnth_5' 'mnth_6' 'mnth_7'
    #  'mnth_8' 'mnth_9' 'mnth_10' 'mnth_11' 'mnth_12' 'holiday_0' 'holiday_1'
    #  'weekday_0' 'weekday_1' 'weekday_2' 'weekday_3' 'weekday_4' 'weekday_5'
    #  'weekday_6' 'workingday_0' 'workingday_1' 'weathersit_1' 'weathersit_2'
    #  'weathersit_3']

    return train_test_split(df, label,
                            test_size=0.25, random_state=1)  # 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
# def featureExtract2():
#     data = pd.read_csv("day.csv")
#     data = data[['season','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']]
#
#     dftrain = data[0:365]
#     dftrain = dftrain.sample(frac=1)
#     dftest = data[365:731]
#     dftest = dftest.sample(frac=1)
#
#     X_train = pd.get_dummies(dftrain, columns=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
#     y_train = X_train[['cnt']]
#     X_train.drop(['cnt'], axis=1, inplace=True)
#     # print(X_train,y_train)
#
#     X_test = pd.get_dummies(dftest, columns=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
#     y_test = X_test[['cnt']]
#     X_test.drop(['cnt'], axis=1, inplace=True)
#
#     # print(X_train.columns.values)
#     # ['temp' 'atemp' 'hum' 'windspeed' 'season_1' 'season_2' 'season_3'
#     #  'season_4' 'mnth_1' 'mnth_2' 'mnth_3' 'mnth_4' 'mnth_5' 'mnth_6' 'mnth_7'
#     #  'mnth_8' 'mnth_9' 'mnth_10' 'mnth_11' 'mnth_12' 'holiday_0' 'holiday_1'
#     #  'weekday_0' 'weekday_1' 'weekday_2' 'weekday_3' 'weekday_4' 'weekday_5'
#     #  'weekday_6' 'workingday_0' 'workingday_1' 'weathersit_1' 'weathersit_2'
#     #  'weathersit_3']
#     return X_train, X_test, y_train, y_test

def try_Model(*data):
    def printScore(s,regr):
        print('------'+s+'------')
        print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
        print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test).reshape(-1,1) - y_test) ** 2))
        print('Score: %.2f' % regr.score(X_test, y_test))
    X_train,X_test,y_train,y_test=data

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    printScore('LinearRegression',regr)

    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    printScore('Ridge',regr)

    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    printScore('Lasso',regr)

    ## 异常值处理
    # error = regr.predict(X_train).reshape(-1,1)-y_train
    # error.abs().to_csv('error2.csv', header = 0)  # 不保存列名

def try_alpha(*data):
    X_train,X_test,y_train,y_test=data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores=[]
    for i,alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Model")
    plt.show()
if __name__=='__main__':

    # featureExtract()
    X_train, X_test, y_train, y_test = featureExtract1()
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    # print(X_test, y_test)

    try_Model(X_train,X_test,y_train,y_test) # 调用 test_Ridge
    # try_alpha(X_train,X_test,y_train,y_test) # 调用 test_Ridge_alpha