# RNN循环神经网络
RNN 是Recurrent Neural Network的简称，作为一种序列模型，用在了很多的地方，如语音识别，机器翻译，视频内容检测等等。由于循环神经网络具有记忆，能够按时序依次处理任意长度的信息，因此在自然语言处理上效果非常好。

[TOC]  
RNN模型  
前向传播
反向传播
LSTM 与 GRU  


## 一、RNN 模型
首先RNN最经典的网络结构图形如下所示：  
![Alt text](images/RNN.png)    
其中，$输入x的序列长度为 T_x，输出y的序列长度为 T_y ，经典RNN输出序列和输入序列长度是一样的，因此T_x=T_y$  
1）对于一个输入为$x$的句子序列，可以细分为一个一个的词，每一个词记为$x^{\langle t \rangle}$，有$x=\{x^{\langle 1 \rangle},x^{\langle 2 \rangle},x^{\langle 3 \rangle}...,x^{\langle t \rangle},...,x^{\langle T_x \rangle}\}$  
2）输出则为同样$y$的句子序列，可以细分为一个一个的词，每一个词记为$y^{\langle t \rangle}$，有$y=\{y^{\langle 1 \rangle},y^{\langle 2 \rangle},y^{\langle 3 \rangle}...,y^{\langle t \rangle},...,y^{\langle T_y \rangle}\}$   
3）除此之外，还包括一个隐状态$a^{[t]}$，每次经过一个$RNN-CELL$预测单元，将会输出$a^{[t]}$。例如$a^{[t-1]}$经过一个事件步后输出$a^{[t]}$，而$y^{\langle T_y \rangle}等于a^{[t]}做一次激活操作获得的结果。$

### 1.1、RNN-CELL  
循环神经网络可以看做是RNN单元的重复，对于RNN-CELL有两种比较好的理解方式：  
1）通过MLP（前馈神经网络）拓扑结构转化：   
一般网络的MLP拓扑结构非常好理解，如下图为神经网络全连接结构：  
![Alt text](images/MLP.png)   
其中输入$x=\{x_1,x_2,x_3,...,x_n\}，输出为y=\{y_1,y_2,...,y_j\}，隐含层为a=\{a_1,a_2,...a_k\}。我们把MLP当做一个RNN-CELL单元，并且将向量x和向量y作为在t时刻的输入输出x^{\langle t \rangle}和y^{\langle t \rangle}。将拓扑结构翻转过来后，做多次拼接，如下图。这样就将RNN和前馈神经网络联系起来了$
![Alt text](images/RNNByMLP.png)  
2）如果以计算单元的方式去理解，可以得到如下所示的运算图形： 
![Alt text](images/RNN-CELL.png)  
运算图形实际上就是展示了RNN单元内部神经网络的实际运算逻辑，对于如上计算单元的实现，可以分为下面几步公式：  
- 通过上一步输入的隐含状态$a^{[t-1]}$计算当前单元的$a^{[t]}$：  
$$a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$$ 
- 通过当前隐含层$a^{[t]}$计算$y^{\langle t \rangle}$  
$$\hat{y}^{\langle t \rangle} = softmax(W_{ya} a^{\langle t \rangle} + b_y)$$  
上面就是RNN的一般前向传播公式，$W_{aa},W_{ax},W_{ya}$为对应所要训练的权值矩阵，$b_a,b_y$是偏置。












