#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/6


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, discriminant_analysis
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
def try_LinearDiscriminantAnalysis(*data):
    '''
    测试 LinearDiscriminantAnalysis 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s'%(lda.coef_,lda.intercept_))
    print('Score: %.2f' % lda.score(X_test, y_test))
def plot_LDA(converted_X,y):
    '''
    绘制经过 LDA 转换后的数据

    :param converted_X: 经过 LDA转换后的样本集
    :param y: 样本集的标记
    :return:  None
    '''
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure()
    ax=Axes3D(fig)
    colors='rgb'
    markers='o*s'
    for target,color,marker in zip([0,1,2],colors,markers):
        pos=(y==target).ravel()
        X=converted_X[pos,:]
        ax.scatter(X[:,0], X[:,1], X[:,2],color=color,marker=marker,
			label="Label %d"%target)
    ax.legend(loc="best")
    fig.suptitle("Iris After LDA")
    plt.show()
def try_LinearDiscriminantAnalysis_shrinkage(*data):
    '''
    测试  LinearDiscriminantAnalysis 的预测性能随 shrinkage 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    shrinkages=np.linspace(0.0,1.0,num=20)
    scores=[]
    for shrinkage in shrinkages:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',
			shrinkage=shrinkage)
        lda.fit(X_train, y_train)
        scores.append(lda.score(X_test, y_test))
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(shrinkages,scores)
    ax.set_xlabel(r"shrinkage")
    ax.set_ylabel(r"score")
    ax.set_ylim(0,1.05)
    ax.set_title("LinearDiscriminantAnalysis")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=txt_strtonum_feed("wifi_localization.txt") # 产生用于分类的数据集
    try_LinearDiscriminantAnalysis(X_train,X_test,y_train,y_test) # 调用 test_LinearDiscriminantAnalysis
    # try_LinearDiscriminantAnalysis_shrinkage(X_train,X_test,y_train,y_test) # 调用 test_LinearDiscriminantAnalysis_shrinkage