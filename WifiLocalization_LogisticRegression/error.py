#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/6


def txt_strtonum_feed(filename):
    with open(filename, 'r') as f:  # with语句自动调用close()方法
        data,label=[],[]
        line = f.readline()
        while line:
            # line.split() 去掉空格并保存为list
            data.append(list(map(int,line.split()[:-1])))
            label.append(line.split()[-1])
            line = f.readline()
    return data,label

data,label = txt_strtonum_feed("wifi_localization.txt")
print(data,label)
l = [[-47.0, -57.0, -52.0, -44.0, -67.0, -78.0, -75.0],
     [-48.0, -58.0, -57.0, -45.0, -73.0, -86.0, -92.0],
     [-45.0, -57.0, -55.0, -48.0, -67.0, -77.0, -84.0],
     [-47.0, -57.0, -57.0, -52.0, -69.0, -79.0, -77.0]]
res = []
for i in range(len(data)):
    for j in range(len(l)):
        if data[i]==l[j]:
            res.append(i+1)
print('res',res)

