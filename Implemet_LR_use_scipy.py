#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/6


import numpy as np
from scipy import optimize


# 读取数据
def txt_strtonum_feed(filename):
    with open(filename, 'r') as f:  # with语句自动调用close()方法
        data, label = [], []
        line = f.readline()
        while line:
            # line.split() 去掉空格并保存为list
            data.append([1] + list(map(float, line.split()[:-1])))
            label.append(1 if line.split()[-1] == '1' else 0)
            line = f.readline()
    return data, label

# 代价函数
def costFunctionReg(theta, X, y, mylambda=1):
    # 样本数量
    m = y.size
    # 参数的拷贝
    tmp_theta = theta.reshape(X.shape[1], 1).copy()
    # 预测函数
    z = np.array(np.dot(X, tmp_theta))
    h = 1.0 / (1.0 + np.e ** -z)
    # 代价函数计算
    J = 1.0 / m * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + 1.0 * mylambda / 2 / m * sum(tmp_theta ** 2)
    # print(J)
    if np.isnan(J):
        return np.inf
    return J

# 梯度
def gradientReg(theta, X, y, mylambda=1):
    # 样本数量
    m = y.size
    # 参数的拷贝
    tmp_theta = theta.reshape(X.shape[1], 1).copy()
    y = y.reshape(X.shape[0],1 ).copy()
    # 预测函数
    z = np.array(np.dot(X, tmp_theta))
    h = 1.0 / (1.0 + np.e ** -z)
    # grad = X.T.dot(h - y)
    grad = 1.0 / m * X.T.dot(h - y) + 1.0 * mylambda / m * tmp_theta
    # 结果变成一行
    # print(X.T.shape)
    # print((h - y).shape)
    # print(grad.ravel().shape)
    # grad = grad.flatten()
    return grad.ravel()

X, y = map(np.array, txt_strtonum_feed("./WifiLocalization_LogisticRegression/wifi_localization.txt"))
initial_theta = np.array([0]*8)
# print(initial_theta.shape)
mylambda=1

result = optimize.minimize(costFunctionReg, initial_theta, args=(X, y, mylambda), method='CG', jac=gradientReg)
theta = result.x
print("\n代价函数计算结果：\n%s"%result.fun)
print("\n参数theta计算结果：\n%s"%result.x)