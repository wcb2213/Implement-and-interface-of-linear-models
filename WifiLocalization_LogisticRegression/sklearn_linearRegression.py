#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/6

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def txt_strtonum_feed(filename):
    with open(filename, 'r') as f:  # with语句自动调用close()方法
        data, label = [], []
        line = f.readline()
        while line:
            # line.split() 去掉空格并保存为list
            data.append(list(map(float, line.split()[:-1])))
            label.append(1 if line.split()[-1] == '1' else 0)
            line = f.readline()
        return train_test_split(data, label,
                                test_size=0.25, random_state=0)  # 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

def try_Ridge(*data):
    '''
    测试 Ridge 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.Lasso(normalize=True)
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))
def try_Ridge_alpha(*data):
    '''
    测试 Ridge 的预测性能随 alpha 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
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
    ax.set_title("Ridge")
    plt.show()
if __name__=='__main__':
    X_train, X_test, y_train, y_test = txt_strtonum_feed("wifi_localization.txt")
    print(X_test,y_test)
    # try_Ridge(X_train,X_test,y_train,y_test) # 调用 test_Ridge
    # try_Ridge_alpha(X_train,X_test,y_train,y_test) # 调用 test_Ridge_alpha