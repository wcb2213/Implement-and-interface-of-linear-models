#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/22


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("day.csv")
# data = data[
#     ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
#      'cnt']]
#
# error_index = [203, 265, 238, 568, 327, 86, 150, 209, 442, 87, 518, 665, 567, 204, 630, 668, 521, 617, 680, 623, 644, 529, 285, 93, 270, 577, 212, 83, 80, 629, 49, 221, 210, 158, 103, 475, 357, 595, 388, 456, 443, 328, 658, 542, 663]
# data.drop(data.index[error_index], inplace=True)
#
# print(data)

df = pd.read_csv("error.csv")
df = df.abs()
print(df)