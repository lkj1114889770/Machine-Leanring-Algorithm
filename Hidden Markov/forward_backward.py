"""
# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 18:48
# @Author  : Kaijian Liu
# @Email   : kaijianliu@qq.com
# @File    : forward_backward.py
# @Software: PyCharm
"""
import numpy as np


def forward(A, B, pi, obs_seq):
    """
    前向算法
    :param A: 概率转移矩阵
    :param B: 观测概率矩阵
    :param pi: 初始状态概率矩阵
    :param obs_seq: 观测序列
    :return: 观测序列概率
    """
    T = len(obs_seq)

    alpha = pi*B[:,obs_seq[0]]
    for t in range(1,T):
        alpha = np.dot(alpha,A)*B[:,obs_seq[t]]

    return np.sum(alpha)


def backward(A, B, pi, obs_seq):
    """
    后向算法
    :param A: 概率转移矩阵
    :param B: 观测概率矩阵
    :param pi: 初始状态概率矩阵
    :param obs_seq: 观测序列矩阵
    :return: 观测序列概率
    """
    T=len(obs_seq)
    N=A.shape[0]
    beta=np.ones((N,1))
    for t in range(T-1,0,-1):
        beta=np.dot(A*B[:,obs_seq[t]],beta)

    return np.sum((pi*B[:,obs_seq[0]]).reshape(-1,1)*beta)


"""
输入的例子以李航《统计学习》 p177页，例10.2为例
"""
A=np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
B=np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
pi=np.array([0.2,0.4,0.4]).T

obs_seq = [0,1,0]

print(forward(A,B,pi,obs_seq))
print(backward(A,B,pi,obs_seq))