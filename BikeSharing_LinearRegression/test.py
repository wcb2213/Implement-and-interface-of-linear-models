#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/22


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def featureExtract():
    data = pd.read_csv("day.csv")
    # print(data.columns.values)
    # ['instant' 'dteday' 'season' 'yr' 'mnth' 'holiday' 'weekday' 'workingday'
    #  'weathersit' 'temp' 'atemp' 'hum' 'windspeed' 'cnt']

    data = data[['season','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']]

    dftrain = data[0:365]
    dftrain = dftrain.sample(frac=1)
    dftest = data[366:732]
    dftest = dftest.sample(frac=1)

    X_train = pd.get_dummies(dftrain, columns=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
    y_train = X_train[['cnt']]
    X_train.drop(['cnt'], axis=1, inplace=True)
    # print(X_train,y_train)

    X_test = pd.get_dummies(dftest, columns=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
    y_test = X_test[['cnt']]
    X_test.drop(['cnt'], axis=1, inplace=True)

    # print(X_train.columns.values)
    # ['temp' 'atemp' 'hum' 'windspeed' 'season_1' 'season_2' 'season_3'
    #  'season_4' 'mnth_1' 'mnth_2' 'mnth_3' 'mnth_4' 'mnth_5' 'mnth_6' 'mnth_7'
    #  'mnth_8' 'mnth_9' 'mnth_10' 'mnth_11' 'mnth_12' 'holiday_0' 'holiday_1'
    #  'weekday_0' 'weekday_1' 'weekday_2' 'weekday_3' 'weekday_4' 'weekday_5'
    #  'weekday_6' 'workingday_0' 'workingday_1' 'weathersit_1' 'weathersit_2'
    #  'weathersit_3']
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = featureExtract()