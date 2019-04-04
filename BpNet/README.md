BP神经网络改变了感知器的结构，引入了新的隐含层以及误差反向传播，基本上能够解决非线性分类问题，也是神经网络的基础网络结构，在此对BP神经网络算法进行总结，并用python对其进行了实现。


BP神经网络的典型结构如下图所示：

![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1508759278442&di=35b034d166ee7a0c6e09c7154c096d3f&imgtype=0&src=http%3A%2F%2Fimgsrc.baidu.com%2Fbaike%2Fpic%2Fitem%2F9922720e0cf3d7ca65c52b8ef01fbe096b63a912.jpg)

隐含层通常为一层，也可以是多层，在BP网络中一般不超过2层。

## 正向传播
正向传播的过程与感知器类似，都是输入与权重的点积，隐含层和输出层都包含一个激活函数，BP网络常用sigmod函数。

![](https://i.imgur.com/5FpsaS2.png)

但是现在好像不常用了，更多地是Tanh或者是ReLU，好像最近又出了一个全新的激活函数，后续还得去了解。
BP神经网络的误差函数是全局误差，将所有样本的误差都进行计算求和，所以在算法过程学习的时候，进行的是批量学习，等所有数据都进行批次计算之后，才进行权重调整。

![](https://i.imgur.com/vW9ndOr.png)

## 反向传播过程
这个可以说是BP网络比较精髓的部分了，也是BP网络能够从数据中学习的关键，误差的反向传播过程就是两种情况，要么输出层神经元，要么是隐含层神经元。

![](https://i.imgur.com/1luEa3W.png)

对于输出神经元，权重的梯度修正法则为：

![](https://i.imgur.com/vc9TjCN.png)

即权重增量等于学习率、局域梯度、输出层输出结果的乘积，对于局域梯度，其计算如下：

![](https://i.imgur.com/xvHCU1P.png)

即为误差信号乘于激活函数的导数，其中n表示第n次迭代。
对于sigmod函数来说，其导数为：

![](https://i.imgur.com/nQVO1tw.png)

对于隐藏层来说，情况更加复杂一点，需要经过上一层的误差传递。

![](https://i.imgur.com/Ep6t5KR.png)

隐藏层的局域梯度为：

![](https://i.imgur.com/pC56jAc.png)

上面式子的第一项，说明隐含层神经元j局域梯度的计算仅以来神经元j的激活函数的导数，但是第二项求和，是上一层神经元的局域梯度通过权重w进行了传递。

总的来说，反向传播算法中，权重的调整值规则为：

（权值调整）=（学习率参数） X （局域梯度） X（神经元j的输入信号）

BP算法中还有一个动量因子（mc），主要是网络调优，防止网络发生震荡或者收敛过慢，其基本思想就是在t时刻权重更新的时候考虑t-1时刻的梯度值。
	
	self.out_wb = self.out_wb + (1-self.mc)*self.eta*dout_wb + self.mc*self.eta*dout_wbold
	self.hi_wb =self.hi_wb + (1-self.mc)*self.eta*dhi_wb + self.mc*self.eta*dhi_wbold

            
输出结果为：

![](https://i.imgur.com/GYv2q53.png)
            
误差输出结果：

![](https://i.imgur.com/xAXcgfX.png)
            
可以看到在1000次左右迭代就已经出现了比较好的结果了。        
            
除了分类，BP神经网络也常用在函数逼近，这时候输出层神经元激活函数一般就不会再采用sigmod函数了，通常采用线性函数。


**【参考文献】**
《神经网络与机器学习》（第3版） （加） Simon Haykin 著；
《机器学习算法原理与编程实践》 郑捷著；
            
            
            
            
            
        
