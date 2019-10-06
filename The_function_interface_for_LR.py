#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/9/25


import numpy as np
from scipy import optimize


def sigmoid(z):
     return 1 / (1 + np.exp(-z))

def costReg(theta, X, y, lamda):
     theta = np.matrix(theta)
     X = np.matrix(X)
     y = np.matrix(y)
     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
     reg = (lamda / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
     return np.sum(first - second) / len(X) + reg

def gradientReg(theta, X, y, lamda):
     theta = np.matrix(theta)
     X = np.matrix(X)
     y = np.matrix(y)

     parameters = int(theta.ravel().shape[1])
     grad = np.zeros(parameters)

     error = sigmoid(X * theta.T) - y

     for i in range(parameters):
          term = np.multiply(error, X[:, i])

          if (i == 0):
               grad[i] = np.sum(term) / len(X)
          else:
               grad[i] = (np.sum(term) / len(X)) + ((lamda / len(X)) * theta[:, i])

     return grad

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def implement_for_LR(X_train, X_test, y_train, y_test,lamda=1):
    n = X_train.shape[1]
    # convert to numpy arrays and initalize the parameter array theta
    X_train = np.array(X_train.values)
    y_train = np.array(y_train.values)
    theta = np.zeros(n)

    result = optimize.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X_train, y_train, lamda))

    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, X_test)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y_test)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))