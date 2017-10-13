# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:49:46 2017

@author: lkj
"""

import pandas as pd
import math

def Tree_building(dataSet):
    tree = []
    if(Calculate_Entropy(dataSet) == 0): #熵为0说明分类已经到达叶子节点
        if(dataSet['play'].sum()==0):  #根据play的值到达0或者1叶子节点
            tree.append(0)
        else:
            tree.append(1)
        return tree
    numSamples=len(dataSet) #样例数
    Feature_Entropy={} #记录按特征A分类后的熵值的字典
    for i in range(1,len(dataSet.columns)-1):
        Set=dict(list(dataSet.groupby(dataSet.columns[i]))) #取出不同的特征
        Entropy=0.0
        for key,subSet in Set.items():
            Entropy+=(len(subSet)/numSamples)*Calculate_Entropy(subSet) #计算熵
        Feature_Entropy[dataSet.columns[i]]=Entropy
    
    #选最小熵值的特征分类点，这样熵值增益最大    
    Feature = min(zip(Feature_Entropy.values(),Feature_Entropy.keys()))[1] 
    Set=dict(list(dataSet.groupby(Feature)))
    for key,value in Set.items():
        subTree=[]
        subTree.append(Feature)
        subTree.append(key)
        subTree.append(Tree_building(value)) #树枝扩展函数的迭代
        tree.append(subTree)
        
    return tree
    
def Calculate_Entropy(data):
    numSamples=len(data)  #样本总数
    P=data.sum()['play']  #正例数量
    N=numSamples-P   #反例数量
    if((N==0)or(P==0)):  
        Entropy=0
        return Entropy
    Entropy = -P/numSamples*math.log(P/numSamples)-N/numSamples*math.log(N/numSamples)
    return Entropy

if __name__ == '__main__':
    data=pd.read_csv('tennis.csv')
    tree=Tree_building(data)
    print(tree)
    