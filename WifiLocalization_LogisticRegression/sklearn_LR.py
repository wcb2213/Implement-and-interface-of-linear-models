#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/6


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import sklearn.datasets
from sklearn.model_selection import train_test_split


def txt_strtonum_feed(filename):
    with open(filename, 'r') as f:  # with语句自动调用close()方法
        data, label = [], []
        line = f.readline()
        while line:
            # line.split() 去掉空格并保存为list
            data.append(list(map(float, line.split()[:-1])))
            label.append(line.split()[-1])
            line = f.readline()
        return train_test_split(data, label,
                                test_size=0.25, random_state=0,stratify=label)  # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
def try_LogisticRegression(*data):
    '''
    测试 LogisticRegression 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))
def try_LogisticRegression_multinomial(*data):
    '''
    测试 LogisticRegression 的预测性能随 multi_class 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))
def try_LogisticRegression_C(*data):
    '''
    测试 LogisticRegression 的预测性能随  C  参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-2,4,num=100)
    scores=[]
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LogisticRegression")
    plt.show()

if __name__=='__main__':
    X_train, X_test, y_train, y_test = txt_strtonum_feed("wifi_localization.txt")
    # X_train, X_test, y_train, y_test = txt_strtonum_feed("wifi_localization_modified.txt")
    try_LogisticRegression(X_train,X_test,y_train,y_test) # 调用  test_LogisticRegression
    # try_LogisticRegression_multinomial(X_train,X_test,y_train,y_test) # 调用  test_LogisticRegression_multinomial
    # try_LogisticRegression_C(X_train,X_test,y_train,y_test) # 调用  test_LogisticRegression_C