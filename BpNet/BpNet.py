# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:12:46 2017

@author: lkj
"""

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

class BpNet(object):
    def __init__(self):
        # 以下参数需要手动设置  
        self.eb=0.01              # 误差容限，当误差小于这个值时，算法收敛，程序停止
        self.eta=0.1             # 学习率
        self.mc=0.3               # 动量因子：引入的一个调优参数，是主要的调优参数 
        self.maxiter=2000         # 最大迭代次数
        self.errlist=[]           # 误差列表
        self.dataMat=0            # 训练集
        self.classLabels=0        # 分类标签集
        self.nSampNum=0             # 样本集行数
        self.nSampDim=0             # 样本列数
        self.nHidden=4           # 隐含层神经元 
        self.nOut=1              # 输出层个数
        self.iterator=0            # 算法收敛时的迭代次数
     
    #激活函数
    def logistic(self,net):
        return 1.0/(1.0+exp(-net))
    
    #反向传播激活函数的导数
    def dlogistic(self,y):
        return (y*(1-y))
    
    #全局误差函数
    def errorfuc(self,x):
        return sum(x*x)*0.5
    
    #加载数据集
    def loadDataSet(self,FileName):
        data=np.loadtxt(FileName)
        m,n=shape(data)
        self.dataMat = np.ones((m,n))
        self.dataMat[:,:-1] = data[:,:-1] #除数据外一列全为1的数据，与权重矩阵中的b相乘
        self.nSampNum = m  #样本数量
        self.nSampDim = n-1  #样本维度
        self.classLabels =data[:,-1]    
    
    #数据集归一化，使得数据尽量处在同一量纲，这里采用了标准归一化
    #数据归一化应该针对的是属性，而不是针对每条数据
    def normalize(self,data):
        [m,n]=shape(data)
        for i in range(n-1):
            data[:,i]=(data[:,i]-mean(data[:,i]))/(std(data[:,i])+1.0e-10)
        return data
    
    #隐含层、输出层神经元权重初始化
    def init_WB(self):
        #隐含层
        self.hi_w = 2.0*(random.rand(self.nSampDim,self.nHidden)-0.5)
        self.hi_b = 2.0*(random.rand(1,self.nHidden)-0.5)
        self.hi_wb = vstack((self.hi_w,self.hi_b))
        
        #输出层
        self.out_w = 2.0*(random.rand(self.nHidden,self.nOut)-0.5)
        self.out_b = 2.0*(random.rand(1,self.nOut)-0.5)
        self.out_wb = vstack((self.out_w,self.out_b))
        
    def BpTrain(self):
        SampIn = self.dataMat
        expected = self.classLabels
        dout_wbold = 0.0
        dhi_wbold = 0.0 #记录隐含层和输出层前一次的权重值，初始化为0
        self.init_WB()
        
        for i in range(self.maxiter):
            #信号正向传播
            #输入层到隐含层
            hi_input = np.dot(SampIn,self.hi_wb)
            hi_output = self.logistic(hi_input)
            hi2out = np.hstack((hi_output,np.ones((self.nSampNum,1))))
            
            #隐含层到输出层
            out_input=np.dot(hi2out,self.out_wb)
            out_output = self.logistic(out_input)
            #计算误差
            error = expected.reshape(shape(out_output)) - out_output
            sse = self.errorfuc(error)
            self.errlist.append(sse)
            if sse<=self.eb:
                self.iterator = i+1
                break
            
            #误差反向传播
            
            #DELTA输出层梯度
            DELTA = error*self.dlogistic(out_output)
            #delta隐含层梯度
            delta =  self.dlogistic(hi_output)*np.dot(DELTA,self.out_wb[:-1,:].T)
            dout_wb = np.dot(hi2out.T,DELTA)
            dhi_wb = np.dot(SampIn.T,delta)
            
            #更新输出层和隐含层权值
            if i==0:
                self.out_wb = self.out_wb + self.eta*dout_wb
                self.hi_wb = self.hi_wb + self.eta*dhi_wb
            else:
               self.out_wb = self.out_wb + (1-self.mc)*self.eta*dout_wb + self.mc*self.eta*dout_wbold
               self.hi_wb =self.hi_wb + (1-self.mc)*self.eta*dhi_wb + self.mc*self.eta*dhi_wbold
            dout_wbold = dout_wb
            dhi_wbold = dhi_wb
    
    ##输入测试点，输出分类结果      
    def BpClassfier(self,start,end,steps=30):
        x=linspace(start,end,steps)
        xx=np.ones((steps,steps))
        xx[:,0:steps] = x
        yy = xx.T
        z = np.ones((steps,steps))
        for i in  range(steps):
            for j in range(steps):
                xi=array([xx[i,j],yy[i,j],1])
                hi_input = np.dot(xi,self.hi_wb)
                hi_out = self.logistic(hi_input)
                hi_out = mat(hi_out)
                m,n=shape(hi_out)
                hi_b = ones((m,n+1))
                hi_b[:,:n] = hi_out
                out_input = np.dot(hi_b,self.out_wb)
                out = self.logistic(out_input)
                z[i,j] = out
        return x,z
                
    def classfyLine(self,plt,x,z):
        #画出分类分隔曲线，用等高线画出
        plt.contour(x,x,z,1,colors='black')
        
    def errorLine(self,plt,color='r'):
        x=linspace(0,self.maxiter,self.maxiter)
        y=log2(self.errlist)
        #y=y.reshape(())
        #print(shape(x),shape(y))
        plt.plot(x,y,color)
        
   # 绘制数据散点图
    def drawDataScatter(self,plt):
        i=0
        for data in self.dataMat:
            if(self.classLabels[i]==0):
                plt.scatter(data[0],data[1],c='blue',marker='o')
            else:
                plt.scatter(data[0],data[1],c='red',marker='s')
            i=i+1
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        