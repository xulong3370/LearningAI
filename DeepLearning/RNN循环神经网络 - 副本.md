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
3）除此之外，还包括一个隐状态$a^{\langle t \rangle}$，每次经过一个$RNN-CELL$预测单元，将会输出$a^{[t]}$。例如$a^{\langle t-1 \rangle}$经过一个事件步后输出$a^{\langle t \rangle}$，而$y^{\langle T_y \rangle}等于a^{\langle t \rangle}做一次激活操作获得的结果。$

### 1.1、RNN-CELL  
循环神经网络可以看做是RNN单元的重复，对于RNN-CELL有两种比较好的理解方式：  
1）通过MLP（前馈神经网络）拓扑结构转化：   
一般网络的MLP拓扑结构非常好理解，如下图为神经网络全连接结构：  
![Alt text](images/MLP.png) 
图（1-1） 普通MLP网络结构 

$其中输入x=\{x_1,x_2,x_3,...,x_n\}，输出为y=\{y_1,y_2,...,y_j\}，隐含层为a=\{a_1,a_2,...a_k\}。我们把MLP当做一个RNN-CELL单元，并且将向量x和向量y作为在t时刻的输入输出x^{\langle t \rangle}和y^{\langle t \rangle}。将拓扑结构翻转过来后，做多次拼接，如下图。这样就将RNN和前馈神经网络联系起来了$
![Alt text](images/RNNByMLP.png)  
图（1-2） 将MLP翻转拼接后得到的RNN结构  

2）如果以计算单元的方式去理解，可以得到如下所示的运算图形：   
![Alt text](images/RNN-CELL.png)  
图（1-3）: 基本的RNN单元  

运算图形实际上就是展示了RNN单元内部神经网络的实际运算逻辑，对于如上计算单元的实现，可以分为下面几步公式：  
- 通过上一步输入的隐含状态$a^{\langle t-1 \rangle}$计算当前单元的$a^{\langle t \rangle}$：  
$$a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a) \tag{1.1}$$ 
- 通过当前隐含层$a^{[t]}$计算$\hat{y}^{\langle t \rangle}$
$$\hat{y}^{\langle t \rangle} = softmax(W_{ya} a^{\langle t \rangle} + b_y) \tag{1.2}$$  
上面就是$RNN$单元的一般前向传播公式，$W_{aa},W_{ax},W_{ya}$为对应所要训练的权值矩阵，$b_a,b_y$是偏置。  
- 假如当前刻输入为$x^{\langle t \rangle}$，他的向量长度为$(n_x)$，上一时间片输入的隐藏层激活值为$a^{\langle t-1 \rangle}$，设置输出长度为$(n_a)$。那么为了保证计算输出的激活值$a^{\langle t \rangle}$的长度不变，$W_{aa}$的维度必须为$(n_a,n_a)$，$W_{ax}$的维度为$(n_a,n_x)$。同理对于输出$\hat{y}^{\langle t \rangle}$，$W_{ya}$的维度为$(n_y, n_a)$。$b_a,b_y$作为偏置维度分别为$(n_a,1)，(n_y,1)$。  

在神经网络运算过程中，如果输入多个样本进行前向传播运算，可以使用for循环的方式。也可以使用矩阵运算的方式，这里一次性输入$m$样本做矩阵运算，则输入$x^{\langle t \rangle}$矩阵变为$(n_x,m)$，$a^{\langle t-1 \rangle}$的维度也变为$(n_x,m)，如下所示为单个RNN单元的前向转播实现：$  


```python
def rnn_cell_forward(xt, a_prev, parameters):
    """
    RNN单元的前向传播
    params：
        xt          当前时间步的输入，一次性计算m个样本，因此维度为(n_x,m)
        a_prev      前一时刻的隐藏状态维度为m个样本的数量（n_a, m）
        parameters  存储整个权重的字典，其中权重有：
                        Wax  维度为（n_a, n_x）与输入xt做矩阵乘法
                        Waa  维度为（n_a, n_a）与前一个隐藏状态a_prev做矩阵乘法
                        Wya  维度为（n_y, n_a）与当前隐藏状态a_next做矩阵乘法
                        ba   维度为（n_a, 1） 偏置
                        by   维度为（n_y, 1） 偏置

    return:
        a_next     当前输出的隐藏状态at，维度（n_a, m）
        yt_pred    当前计算的输出，维度（n_y， m）
        cache      用于反向传播需要的元组集合，(a_next, a_prev, xt, parameters)
    """
    # 从“parameters”获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # 完成公式（1.1）的计算
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    
    # 完成公式（1.2）的计算
    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)
    
    # 保存反向传播需要的值
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache
```

### 1.2 RNN前向传播计算（对多个单元进行连接）
对如上RNN-CELL进行重复拼接得到RNN模型   

![Alt text](images/RNN-Multi.png)  
图（1-3）: RNN单元重复拼接而成的基本模型，也可以理解成图（1-2）MLP翻转拼接的神经网络结构的计算图  

因此可以实现RNN前向转播，示例代码如下：
```python
def rnn_forward(x, a0, parameters):
    """
    拼接RNN单元实现RNN的整体前向传播
    
    params：
        x           将所有xt时间输入拼接到一个矩阵中，一次性计算m个样本，因此维度为(n_x,m,T_x)
        a0          时间步为1的单元输入的隐藏状态是不存在的，因此对其初始化，维度（n_a, m）
        parameters  存储整个权重的字典，其中权重有：
                        Wax  维度为（n_a, n_x）与输入xt做矩阵乘法
                        Waa  维度为（n_a, n_a）与前一个隐藏状态a_prev做矩阵乘法
                        Wya  维度为（n_y, n_a）与当前隐藏状态a_next做矩阵乘法
                        ba   维度为（n_a, 1） 偏置
                        by   维度为（n_y, 1） 偏置
    
    返回：
        a          所有时间步的隐藏状态，维度为(n_a, m, T_x) -> [a1,a2,a3,...,at]
        y_pred     所有时间步的预测，维度为(n_y, m, T_x) -> [y1,y2,y3,...,yt]
        caches     为反向传播的保存的元组
    """
    
    # 使用caches 去装载所有cache
    caches = []
    
    # 获取 x 与 Wya 的维度信息，用于构造a和y
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # 直接采用0矩阵来初始化a与y
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])
    
    # 使用外部输入的初始激活值a0进行计算
    a_next = a0
    
    # 使用for循环遍历所有时间
    for t in range(T_x):
        # 取出当前时间x[:, :, t]，通过cell计算下一步的激活值a和预测值y，并保留cache
        # cache 是一个 (a_next, a_prev, xt, parameters) 保留所有信息的元祖
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        
        # 将t当前a值存起来
        a[:, :, t] = a_next
        
        # 将t当前y值存起来
        y_pred[:, :, t] = yt_pred
        
        # 把cache保存到caches列表中。
        caches.append(cache)
    
    # 全部完成后，把x输入矩阵也全部加入进来，实际上cache中已经存了每个时间步的x了
    caches = (caches, x)
    
    return a, y_pred, caches
```

### 1.3 RNN反向转播的计算
RNN反向传播跟一般的神经网络模型有一些区别，就是需要做递归操作，这里先看如下图，之后再慢慢解释：  

![Alt text](images/RNN-backward.png)   
图（1-4）: RNN单元的反向传播

对于RNN，我们在序列的每个单元都存在损失函数，我们把损失函数定义为$J$。训练本质就是求$(W_{ax},W_{aa},b_a,W_{ya},b_y)$合适的值（注意在序列每个CELL中，这些值都是共用的，即RNN序列共用一套权重），采用梯度下降法求他们的梯度并更新这些值。  
RNN对于序列输入 $\{x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, ..., x^{\langle T_x \rangle}\}$,  会有序列输出 $\{y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., x^{\langle T_y \rangle}\}$。最后的损失函数$J$是把各个时间点$y^{\langle t \rangle}$的单个 $J^{\langle t \rangle}$ 加起来。根据导数的性质（和的导数等于导数的和），可以化解成对每个$J^{\langle t \rangle}$求导，最后加起来。    

首先对于某一个单元可知：  
$$ J^{\langle t \rangle}=Loss(\hat{y}^{\langle t \rangle})   \tag{1.3}$$   

对于$(W_{ya},b_y)$求取梯度比较简单，说明下公式假定输入输出一样$T_x=T_y=τ$，根据（1.2）式我们可以先拆分得到如下两个式子，用$o^{\langle t \rangle}$ 做线程输出的赋值：  
$$\hat{y}^{\langle t \rangle} = softmax(o^{\langle t \rangle}) \tag{1.4}$$    
$$o^{\langle t \rangle} =W_{ya} a^{\langle t \rangle} + b_y \tag{1.5}$$     

根据公式（1.3）、（1.4）、（1.5）通过链式法则得到：  
$$\frac{\partial J}{\partial W_{ya}}=\quad \sum_{t=1}^{τ}\frac{\partial J^{\langle t \rangle}}{\partial W_{ya}}=\quad \sum_{t=1}^{τ}\frac{\partial J^{\langle t \rangle}}{\partial o^{\langle t \rangle}} \frac{\partial o^{\langle t \rangle}}{\partial W_{ya}} \tag{1.4}  $$   
$$\frac{\partial J}{\partial b_y}=\quad \sum_{t=1}^{τ}\frac{\partial J^{\langle t \rangle}}{\partial b_y}=\quad \sum_{t=1}^{τ}\frac{\partial J^{\langle t \rangle}}{\partial o^{\langle t \rangle}} \frac{\partial o^{\langle t \rangle}}{\partial b_y} \tag{1.5}  $$   

为什么要用 $\frac{\partial J}{\partial o}$ 而不用$\frac{\partial J}{\partial y}$ 呢？因为损失函数可以有很多形式，一般损失函数对于输出$o$求$\frac{\partial J}{\partial o}$都有相应的推导公式，这里举例如果使用的是交叉熵函数：  
$$J^{\langle t \rangle}=\quad -\sum_{i=1}^{n_y}y_i^{\langle t \rangle}\log \hat{y_i}^{\langle t \rangle} \tag{1.6}$$  
$其中y_i^t为真实值，\hat{y_i}^{\langle t \rangle}为预测值,i为t时间片输出的向量i位置值$。

这里不再推导交叉熵的倒数公式，有兴趣的可以查看相应文章去推导，通过交叉熵公式可以得到$J$对$o$的倒数如下： 
$$\frac{\partial J^{\langle t \rangle}}{\partial o^{\langle t \rangle}}=\hat{y}^{\langle t \rangle}- y^{\langle t \rangle} \tag{1.7}$$
那么（1.4）和（1.5）式分别变为：  
$$ \frac{\partial J}{\partial W_{ya}}=\quad \sum_{t=1}^{τ}(\hat{y}^{\langle t \rangle}- y^{\langle t \rangle})(a^{\langle t \rangle })^T \tag{1.8}$$  
$$ \frac{\partial J}{\partial b_y}=\quad \sum_{t=1}^{τ}(\hat{y}^{\langle t \rangle}- y^{\langle t \rangle}) \tag{1.9}$$  
注意$(a^{\langle t \rangle })^T中T是矩阵的转置，矩阵的求导法则一般是用迹计算，乘函数AB的迹对A求导 结果等于矩阵B的转置，公式如下$。
$$\frac{\partial tr(AB)}{\partial A} = B^T \tag{1.10}$$ 

而对于$(W_{ax},W_{aa},b_a)$稍微复杂一点，需要引入一个概念BPTT（back-propagation through time）的运算法则，公式如下：  

。。。。


## 二、LSTM 长短记忆模型  
传统的RNN反向传播拥有如下特点：
- Tanh 输出在-1和1之间
- 梯度消失
- 较远的步骤梯度贡献很小
- 切换其他激活函数后，可能也会导致梯度爆炸  
  
为什么需要LSTM：  
- 普通的RNN的信息不能长久传播（存在于理论上）
- 引入选择性机制（选择性输出、选择性输入、选择性遗忘）
- 选择性用门阀控制，使用Sigmoid函数[0,1]







