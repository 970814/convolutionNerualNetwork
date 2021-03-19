
## 卷积神经网络

### 应用一

##### 手写数字识别




   <img width="249" alt="" src="https://tf.wiki/_images/mnist_0-9.png">

- 达到的效果
  - 拟合度99.707%,测试集上 准确率**99.12%** ，在训练了9个epoth获得
  - 仅仅训练了9个epoth，相比于全连接神经网络已经提升了近1个百分点，是很大的改进
  - 实际上，通过一些优化方法的组合，如 _数据增强、合理调整卷积层卷积核个数大小、隐藏层神经元个数、BatchNormalize、L2 Regularization、Dropout_ 等方法， 是能够将模型的准确率提升至99.6%的。这些优化待更新。

- 使用[mnist训练集](http://neuralnetworksanddeeplearning.com/chap4.html)
  - 训练集为50000张图片
  - 测试集为10000张图片


- 学习曲线

训练集上cost 关于 迭代次数 的变化曲线

<img width="496" alt="" src="https://user-images.githubusercontent.com/19931702/111852337-844e1c80-8951-11eb-99e3-e966dbac10da.png">

训练集和测试上的准确率关于 迭代epoths次数的变化曲线

<img width="496" alt="" src="https://user-images.githubusercontent.com/19931702/111852398-b9f30580-8951-11eb-81c6-1bde39de8686.png">


训练集和测试上的cost关于 迭代epoths次数的变化曲线

<img width="497" alt="" src="https://user-images.githubusercontent.com/19931702/111852418-c8d9b800-8951-11eb-82ef-4e20acbfb706.png">

训练每个epoth所花费的时间曲线变化

<img width="503" alt="" src="https://user-images.githubusercontent.com/19931702/111852436-e3ac2c80-8951-11eb-818b-f34aa29c29e7.png">


### 测试
- 运行**testCNNGradient.m**文件，该算法使用反向传播算法计算
一个具有多通道、3卷积层、3池化层、2个全连接层、一个softmax层的网络的梯度，
  并使用双侧差分方法近似网络梯度来验证。如果一切正确，将完成对4948 个w的偏导数验证，79 个b的偏导数验证。
  
### 算法
1. 向前传播算法计算cost，详细查看文件 **forwardPropagation.m**
   - 算法进行了优化，能在0.3s左右对100个mnist数据进行预测
2. 反向传播算法计算梯度，详细查看文件 **backPropagation.m**
   - 算法进行了优化，能在6s左右对100个mnist数据计算梯度


### 实现&运行 环境
1. 编程语言
octave6.1.0
2. 操作系统
macOS Big Sur，芯片 Apple M1，内存 16 GB




#### 主要参考资料

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) 

