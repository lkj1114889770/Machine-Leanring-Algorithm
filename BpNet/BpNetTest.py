# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:08:28 2017

@author: lkj
"""

from BpNet import *
import matplotlib.pyplot as plt 

# 数据集
bpnet = BpNet() 
bpnet.loadDataSet("testSet2.txt")
bpnet.dataMat = bpnet.normalize(bpnet.dataMat)

# 绘制数据集散点图
bpnet.drawDataScatter(plt)

# BP神经网络进行数据分类
bpnet.BpTrain()

print(bpnet.out_wb)
print(bpnet.hi_wb)

# 计算和绘制分类线
x,z = bpnet.BpClassfier(-3.0,3.0)
bpnet.classfyLine(plt,x,z)
plt.show()
# 绘制误差曲线
bpnet.errorLine(plt)
plt.show()
