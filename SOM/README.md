---
title: 自组织特征映射神经网络（SOM）
date: 2017-09-30 10:39:06
tags:
	- 机器学习
	- 无监督学习
---
自组织特征映射神经网络（SOM）设计思想基于人体大脑，外界信息输入时，大脑皮层对应区域的部分神经元会产生兴奋，位置临近的神经云也会有相近的刺激。大脑神经元的这种特点，不是先天安排好的，而是通过后天的自学习组织形成的。芬兰Helsinki大学的Kohonen教授提出了一种成为自组织特征映射的神经网络模型。
<!-- more -->

## SOM介绍
SOM与kmeans算法有点相似，其基本思想也是，将距离小的个体集合划分为同一类别，而将距离大的个体划分为不同的类别。

![SOM结构图](http://appwk.baidu.com/naapi/doc/view?ih=584&o=png_6_0_0_436_680_396_210_876_1252.5&iw=1100&ix=0&iy=414&aimw=1100&rn=1&doc_id=d6e8e02c647d27284b735162&pn=1&sign=b919aed799fa017c1e7f322b49ebb944&type=1&app_ver=2.9.8.2&ua=bd_800_800_IncredibleS_2.9.8.2_2.3.7&bid=1&app_ua=IncredibleS&uid=&cuid=&fr=3&Bdi_bear=WIFI&from=3_10000&bduss=&pid=1&screen=800_800&sys_ver=2.3.7)

从结构上看，SOM比较简单，只有两层，输入层和竞争层，只有输入层到竞争层的权重向量需要训练，不同于其他神经网络，竞争层同层之间的神经元还有侧向连接，在学习的过程中还会相互影响。竞争层神经元的竞争通过神经元对应的权重向量和输入样本的距离比较，距离最近的神经元成为获胜节点。

常见的相互连接的调整方式有以下几种
1. 墨西哥草帽函数：获胜节点，有最大的权值调整量，离获胜节点越远，调整量越小，甚至达到某一距离，调整量还会变为负值。如图a所示。
2. 大礼帽函数：墨西哥草帽函数的简化，如图b所示。
3. 厨师帽函数：大礼帽函数的简化，如图c所示。

![](https://i.imgur.com/JtGnkfW.png)

## SOM算法学习过程

### 网络初始化
输入层网络节点数与输入样本维度（列数）相同，通常需要进行数据归一化，常见的方法是标准归一化。
竞争层网络根据数据维度以及分类类别数来确定，二维数据和4种分类的话
，竞争层含4个节点，但是权重矩阵为4X2，权重矩阵的初始化一般·随机给一个0-1之间的随机值。对于分类类别数目不清楚的情况，可以定义竞争层多于可能的实际分类的节点，这样最后训练结果中不对应分类结果的节点始终不会收到刺激而兴奋，即抑制。

学习率会影响收敛速度，一般定义一个动态的学习率，随迭代次数增加而递减。优胜邻域半径也定义为一个动态收缩的，随着迭代次数增加而递减。

	# 学习率和学习半径函数	
	def ratecalc(self,indx):
		lrate = self.lratemax-(float(indx)+1.0)/float(self.Steps)*(self.lratemax-self.lratemin) 
		r = self.rmax-(float(indx)+1.0)/float(self.Steps)*(self.rmax-self.rmin)
		return lrate,r

### 网络训练
训练过程如下：
1. 随机抽取一个数据样本，计算竞争层中神经元对应的权重矩阵与数据样本的距离，找到距离最近的为获胜节点。
2. 根据优胜邻域半径，找出此邻域内的所有节点。
3. 根据学习率调整优胜邻域半径内的所有节点，然后回到步骤1进行迭代，直到到达相应的迭代次数
4. 根据最终的迭代结果，为分类结果分配标签。

	# 主算法	
	def train(self):
    #1 构建输入层网络
		dm,dn = shape(self.dataMat) 
		normDataset = self.normalize(self.dataMat) # 归一化数据x
		#2 构建分类网格
		grid = self.init_grid() # 初始化第二层分类网格 
		#3 构建两层之间的权重向量
		self.w = random.rand(dn,self.M*self.N); #随机初始化权值 w
		distM = self.distEclud   # 确定距离公式
		#4 迭代求解
		if self.Steps < 10*dm:	self.Steps = 10*dm   # 设定最小迭代次数
		for i in range(self.Steps): 	
			lrate,r = self.ratecalc(i) # 计算学习率和分类半径
			self.lratelist.append(lrate);self.rlist.append(r)
			# 随机生成样本索引，并抽取一个样本
			k = random.randint(0,dm) 
			mySample = normDataset[k,:] 	
	
			# 计算最优节点：返回最小距离的索引值
			minIndx= (distM(mySample,self.w)).argmin()
			d1 = int(round(minIndx - r));
			d2 = int(round(minIndx + r));
			
			if(d1<0): 
				d1=0
			if(d2>(shape(self.w)[1]-1)):
				d2= shape(self.w)[1]-1
			di=d1;
			#print(d1,d2)
			while(di<=d2):
				self.w[:,di] = self.w[:,di]+lrate*(mySample[0]-self.w[:,di])
				di=di+1
                
		# 分配类别标签
		for i in range(dm):
			self.classLabel.append(distM(normDataset[i,:],self.w).argmin())
		self.classLabel = mat(self.classLabel)		

具体实现代码可见：[https://github.com/lkj1114889770/Machine-Leanring-Algorithm/tree/master/SOM](https://github.com/lkj1114889770/Machine-Leanring-Algorithm/tree/master/SOM)

针对数据集的分类结果：

![](https://i.imgur.com/KSlpNem.png)


**
参考文献：**
1. 《机器学习算法与编程实践》 郑捷著；


