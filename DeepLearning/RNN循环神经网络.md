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
对于一个输入为$x$的句子序列，可以细分为一个一个的词，每一个词记为$x^{\langle t \rangle}$，有$x=\{x^{\langle 1 \rangle},x^{\langle 2 \rangle},x^{\langle 3 \rangle}...,x^{\langle t \rangle},...,x^{\langle T_x \rangle}\}$  
输出则为同样$y$的句子序列，可以细分为一个一个的词，每一个词记为$x^{\langle t \rangle}$，有$y=\{y^{\langle 1 \rangle},y^{\langle 2 \rangle},y^{\langle 3 \rangle}...,y^{\langle t \rangle},...,y^{\langle T_y \rangle}\}$   
除此之外，还包括一个隐状态$a^{[t]}$，每次进过一个$RNN-CELL$预测单元，将会更新这个权重。这个可以理解为记忆状态，前面词和后面词的一种关联。










