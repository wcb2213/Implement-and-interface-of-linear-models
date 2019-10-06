#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/22


n,m,k=map(int,input().split(' '))
input_list = []
for i in range(k):
    input_list.append(list(map(int, input().split(' '))))

flag = [0]*m
for i in range(k):
    tmp_list = input_list[i][1:]
    min_num = min(tmp_list)
    flag[min_num-1]=min_num
    for j in tmp_list:
        flag[j-1] = min_num

print(len(set(flag))-1)