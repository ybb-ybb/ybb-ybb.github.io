---
date: 2020-03-4 11:38:00
description: NIPS2018最佳论文
title: neural ode
author: 鱼摆摆
comments: true
tags: 
 - neural ode
photos: https://cn.bing.com/th/id/OIP.eHoWrQABrthZZb_QHslOLQHaKe?pid=Api&rs=1
categories: 学习
---

原文：[Neural ordinary differential equations](https://arxiv.org/abs/1806.07366v2 )

NIPS2018最佳论文

# 简介

简单复习一下ResNet：

![image.png](https://i.loli.net/2020/02/27/zRXLIKf7rwB4DMJ.png)

通过残差块解决反向传播过程中的梯度消失问题

ResNet、RNN、Normalizing flow 等模型都是这种形式：
$$
h_{t+1}=h_t+f(h_t,\theta_t)
$$
如果采用更多的层数和更小的步长，可以优化为一个常微分方程：
$$
\frac{d \mathbf{h}(t)}{d t}=f(\mathbf{h}(t), t, \theta)
$$

这就是ODE Net的核心idea了……下面进行具体的分析



给定常微分方程，数学理论上可以对其进行解析法求解，但通常我们只关心数值解：在已知$h(t_0)$ 的情况下，求出$h(t_1)$ 。这在神经网络里对应的是正向传播。用ResNet对比一下：

ResNet的正向传播：
$$
h_{t+1}=h_t+f(h_t,\theta_t)
$$
ODE网络的正向传播：

$$
\begin{array}{c}
\frac{d h(t)}{d t}=f(h(t), t, \theta) \\
\int_{t_{0}}^{t_{1}} d h(t)=\int_{t_{0}}^{t_{1}} f(h(t), t, \theta) d t \\
h\left(t_{1}\right)=h\left(t_{0}\right)+\int_{t_{0}}^{t_{1}} f(h(t), t, \theta) d t
\end{array}
$$

求解这个常微分方程数值解的方法有很多，最原始的是欧拉法：固定$\Delta t$ ,通过逐步迭代来求解：
$$
h(t+\Delta t)=h(t)+\Delta t * f(h(t),t,\theta)
$$
我们看到，如果令$\Delta t=1$ ,离散化的欧拉法就退化成残差模块的表达式，也就是说ResNet可以看成是ODENet的特殊情况。 但欧拉法只是解常微分方程最基础的解法，它每走一步都会产生误差，并且误差会层层累积起来。近百年来，在数学和物理学领域已经有更成熟的ODE Solve方法，它们不仅能保证收敛到真实解，而且能够控制误差，本文在不涉及ODE Solve内部结构的前提下(将ODE Solve作为一个黑盒来使用)，研究如何用ODE Solve帮助机器学习。

这篇文章使用了一种适应性的ODE solver，它不像ResNet那样固定步长，而是根据给定的误差容忍度自动调整步长，黑色的评估位置可以视作神经元，他的位置也会根据误差容忍度自动调整：

![image.png](https://i.loli.net/2020/02/27/vcz59MYDShBRdmA.png)

使用ODENet的几个好处（和原文不完全一致，详细可看原文）：

- 一般的神经网络利用链式法则，将梯度从最外层的函数逐层向内传播，并更新每一层的参数$\theta$ ,这就需要在前向传播中需要保留所有层的激活值，并在沿计算路径反传梯度时利用这些激活值。这对内存的占用非常大，层数越多，占用的内存也越大，这限制了深度模型的训练过程。 本文给出的用ODENet反向传播的方法不存储任何中间过程，因而不管层数如何加深，只需要常数级的内存成本。
- 自适应的计算。传统的欧拉法会有误差逐层累积的缺陷，而ODENet可以在训练过程中实时的监测误差水平，并可以调整精度来控制模型的成本。例如：在训练时我们可以使用较高的精度使训练的模型尽可能准确，而在测试时可以使用较低的精度，减少测试成本。
- 应用在流模型上会极大简化变分公式的计算，在下文中详细讲解
- 在时间上的连续性，好理解不展开

对于ODEnet在流模型上的应用，可以看一下论文FFJORD。

# 反向传播

在训练连续神经网络的过程中，正向传播可以使用ODE slove。但对ODE solve求导来进行反向传播求解梯度是很困难的，本篇文章使用Pontryagin的伴随方法(adjoint method) 来求解梯度，该方法不仅在计算和内存上有更大优势，同时还能够精确地控制数值误差。

具体而言，对于：
$$
\left.L\left(\mathbf{z}\left(t_{1}\right)\right)=L\left(\int_{t_{0}}^{t_{1}} f(\mathbf{z}(t), t, \theta) d t\right)=L\left(\text { ODESolve(z }\left(t_{0}\right), f, t_{0}, t_{1}, \theta\right)\right)\tag{1}
$$

为优化$L$ ,我们需要计算他对于参数$z(t_0),t_0,t_1$ 和$\theta$ 的梯度。

第一步是确定loss的梯度如何取决于隐藏状态$z(t)$ 的变化，这在文章中被称作伴随$a(t)$ ($adjoint \quad a(t)$ )
$$
a(t) = - \partial L / \partial \mathbf{z}(t)
$$

这个$a(t)$ 实际等价于反向传播算法中的梯度，可以由另一个ODE给定(证明补充在后面)：
$$
\frac{d a(t)}{d t}=-a(t)^{\top} \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}}\tag{2}
$$
在传统的基于链式法则的反向传播过程中，我们将后一层对前一层进行求导以传递梯度($\partial L/\partial z(t_0)=\partial L/\partial z(t_1) * \partial z(t_1) / \partial z(t_0)$)，而在ODENet中，可以再次调用ODESolve计算$\partial L/\partial z(t_0)$。

对于计算相对于参数$\theta$ 的梯度，公式类似：
$$
\frac{d L}{d \theta}=\int_{t_{1}}^{t_{0}} a(t)^{\top} \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \theta} d t\tag{3}
$$
这三个积分(带标号的三个)可以在同一个ODE solver过程中进行计算：

![image.png](https://i.loli.net/2020/02/27/BkJK7O1DlC2feSX.png)



简单解释如何理解上面这个算法：

以前向传播为例，$ODESolve(h(t_0),f,t_0,t_1,\theta)$ 表示求解常微分方程$\frac{d \mathbf{h}(t)}{d t}=f(\mathbf{h}(t), t, \theta)$ 的数值解$h(t_1)$ 。

将积分串联在一起以使用一次ODESolver解出所有量。

如果Loss不仅仅取决于最终状态，那么在用ODENet进行反向传播时，需要在这些状态中进行一系列的单独求解，每次都要调整算法中的 adjoint $a(t)$ 。

![image.png](https://i.loli.net/2020/02/29/eEYTBROQXak9zGm.png)

# 用ODE网络替代ResNet

对图像进行两次下采样，然后分别应用六个残差块和一个ODENet进行对比：

![](https://cdn.mathpix.com/snip/images/9PA_hUHivtHxmQa6DjIyIrlEMtVj-DmZI_0EuLBgbfU.original.fullsize.png)

RK-Net是用Runge-Kutta积分器，直接进行反向误差的传播

$L$ 表示ResNet的隐藏层数，$\tilde{L}$ 表示调用ODESolve的次数。

误差控制，前向传播和反向传播的求值次数，网络深度表现在下图：

![image.png](https://i.loli.net/2020/02/29/hOvQMLZalHcYbK6.png)

a：求值次数和精度成反比

b：求值次数与时间成正比

c：求值次数与反向传播时间成正比，并且反向传播的时间大概是正向传播的一半

d：网络深度，由于ODENet是一个连续网络，没有隐藏层，因此将评估点的数量作为深度，可以看到在训练过程中网络深度逐渐增加

# 连续的归一化流模型

流模型的解读在前一篇博客中

流模型使用一个可逆函数$f$ 进行两个分布之间的映射，变换前后的两个分布满足变量代换定理：
$$
\mathrm{z}_{1}=f\left(\mathrm{z}_{0}\right) \Longrightarrow \log p\left(\mathrm{z}_{1}\right)=\log p\left(\mathrm{z}_{0}\right)-\log \left|\operatorname{det} \frac{\partial f}{\partial \mathrm{z}_{0}}\right|
$$
平面归一化流(NICE之后的一篇流模型的文章，这篇论文没看……)使用的变换：
$$
\mathbf{z}(t+1)=\mathbf{z}(t)+u h\left(w^{\top} \mathbf{z}(t)+b\right), \quad \log p(\mathbf{z}(t+1))=\log p(\mathbf{z}(t))-\log \left|1+u^{\top} \frac{\partial h}{\partial \mathbf{z}}\right|
$$
在流模型中，为使$\partial f/\partial z$ 的雅可比行列式易于计算，通常是通过精心构建函数$f$ 来实现。并且$f$ 还需要是可逆的。而在这篇文章里发现，将离散的流模型换成连续流模型，可以极大的简化计算：不需要去计算$\partial f/ \partial z$ 的行列式，只需要计算迹，并且不需要构建$f$ 可逆：$f$ 可以是任意函数，它是天然可逆的(常微分方程决定的函数只要满足唯一性，就一定是双射的)，因此$f$ 理论上可以是任何网络。

核心定理（梯度变元定理）：

![image.png](https://i.loli.net/2020/03/01/vGXgdcak5LtE3Fy.png)

证明过程见附录

于是我们将平面归一化流模型连续化：
$$
\frac{d \mathbf{z}(t)}{d t}=u h\left(w^{\top} \mathbf{z}(t)+b\right), \quad \frac{\partial \log p(\mathbf{z}(t))}{\partial t}=-u^{\top} \frac{\partial h}{\partial \mathbf{z}(t)}
$$
不同于求行列式的值，求迹还是一个连续函数，因此如果常微分方程$dz/dt$ 是由一组函数的和给出的，那么对数概率密度也可以直接用迹的和表示：
$$
\frac{d \mathbf{z}(t)}{d t}=\sum_{n=1}^{M} f_{n}(\mathbf{z}(t)), \quad \frac{d \log p(\mathbf{z}(t))}{d t}=\sum_{n=1}^{M} \operatorname{tr}\left(\frac{\partial f_{n}}{\partial \mathbf{z}}\right)
$$
因此对于有M个隐藏状态的连续流模型来说，计算成本仅仅是$\mathcal{O}\left(M\right)$,而平面归一化流的计算成本是$\mathcal{O}\left(M^{3}\right)$ 。

NF和CNF的比较：

![image.png](https://i.loli.net/2020/03/01/q2hVZMcYfgFe4yQ.png)

![image.png](https://i.loli.net/2020/03/01/78uardRHgZjOmJw.png)

# 通过ODE对时间序列建模

$$
z_{t_{0}} \sim p\left(z_{t_{0}}\right)
$$

$$
z_{t_{1}}, z_{t_{2}}, \ldots, z_{t_{N}} =ODESolve(z_{t_0},f,\theta_f,t_0,\ldots,t_N)
$$

$$
each \quad x_{t_i} \sim p(x|z_{t_i},\theta_X)
$$

具体而言，在给定初始状态 $z_0$ 和观测时间 $t_0,\ldots t_N$ 的情况下，该模型计算潜在状态 $z_{t_1} \ldots z_{t_N}$ 和输出 $x_{t_1} \ldots x_{t_N}$。在实验部分，初始状态$z_0$由RNN编码产生，潜在状态$z_{t_1} \ldots z_{t_N}$ 由ODESolve产生，其中的$f$ 用神经网络训练，然后利用VAE的方式从潜在状态中生成数据。

![image.png](https://i.loli.net/2020/03/01/4DVZymwkpOnMHEC.png)

实验：从采样点进行螺旋线重建

![image.png](https://i.loli.net/2020/03/01/tv6yNBm9xe8wIoV.png)

均方差比较：

![image.png](https://i.loli.net/2020/03/01/VDPwMltRHsAG3hf.png)

# 附录

## 伴随法的证明

### 对于$z(t)$

给定常微分方程：
$$
\frac{dz(t)}{d(t)}=f(z(t),t,\theta)
$$

$$
\left.L\left(\mathbf{z}\left(t_{1}\right)\right)=L\left(\int_{t_{0}}^{t_{1}} f(\mathbf{z}(t), t, \theta) d t\right)=L\left(\text { ODESolve(z }\left(t_{0}\right), f, t_{0}, t_{1}, \theta\right)\right)
$$

定义便随状态：
$$
a(t) = - \partial L / \partial \mathbf{z}(t)
$$

则：
$$
\frac{d a(t)}{d t}=-a(t)^{\top} \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}}\tag{2}
$$
证明：

$$
\mathbf{z}(t+\varepsilon)=\int_{t}^{t+\varepsilon} f(\mathbf{z}(t), t, \theta) d t+\mathbf{z}(t)=T_{\varepsilon}(\mathbf{z}(t), t)
$$
然后应用链式法则，有：
$$
\frac{d L}{\partial \mathbf{z}(t)}=\frac{d L}{d \mathbf{z}(t+\varepsilon)} \frac{d \mathbf{z}(t+\varepsilon)}{d \mathbf{z}(t)} \quad \text { or } \quad \mathbf{a}(t)=\mathbf{a}(t+\varepsilon) \frac{\partial T_{\varepsilon}(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)}
$$
 利用导数定义，并进行泰勒展开进行化简计算：
$$
\begin{aligned}
\frac{d \mathbf{a}(t)}{d t} &=\lim _{\varepsilon \rightarrow 0^{+}} \frac{\mathbf{a}(t+\varepsilon)-\mathbf{a}(t)}{\varepsilon} \\
&=\lim _{\varepsilon \rightarrow 0^{+}} \frac{\mathbf{a}(t+\varepsilon)-\mathbf{a}(t+\varepsilon) \frac{\partial}{\partial \mathbf{z}(t)} T_{\varepsilon}(\mathbf{z}(t))}{\varepsilon} \\
&=\lim _{\varepsilon \rightarrow 0^{+}} \frac{\mathbf{a}(t+\varepsilon)-\mathbf{a}(t+\varepsilon) \frac{\partial}{\partial \mathbf{z}(t)}\left(\mathbf{z}(t)+\varepsilon f(\mathbf{z}(t), t, \theta)+\mathcal{O}\left(\varepsilon^{2}\right)\right)}{\varepsilon} \\
&=\lim _{\varepsilon \rightarrow 0^{+}} \frac{\mathbf{a}(t+\varepsilon)-\mathbf{a}(t+\varepsilon)\left(I+\varepsilon \frac{\partial f(\mathbf{z}(t), t, \theta)}{\theta \mathbf{z}(t)}+\mathcal{O}\left(\varepsilon^{2}\right)\right)}{\varepsilon} \\
&=\lim _{\varepsilon \rightarrow 0^{+}} \frac{-\varepsilon \mathbf{a}(t+\varepsilon) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}+\mathcal{O}\left(\varepsilon^{2}\right)}{\varepsilon} \\
&=\lim _{\varepsilon \rightarrow 0^{+}}-\mathbf{a}(t+\varepsilon) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}+\mathcal{O}(\varepsilon) \\
&=-\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}
\end{aligned}
$$

### 对于$\theta$ 和$t$

定义增强状态：
$$
\frac{d}{d t}\left[\begin{array}{l}
\mathrm{z} \\
\theta \\
t
\end{array}\right](t)=f_{\operatorname{aug}}([\mathrm{z}, \theta, t]):=\left[\begin{array}{c}
f([\mathrm{z}, \theta, t]) \\
0 \\
1
\end{array}\right], \mathbf{a}_{a u g}:=\left[\begin{array}{l}
\mathrm{a} \\
\mathrm{a}_{\theta} \\
\mathrm{a}_{t}
\end{array}\right], \mathrm{a}_{\theta}(t):=\frac{d L}{d \theta(t)}, \mathrm{a}_{t}(t):=\frac{d L}{d t(t)}
$$
其中$\theta$ 和$t$ 无关，即$d\theta(t) /dt=0,dt(t)/dt=1$

计算雅可比行列式：
$$
\frac{\partial f_{a u g}}{\partial[\mathbf{z}, \theta, t]}=\left[\begin{array}{ccc}
\frac{\partial f}{\partial z} & \frac{\partial f}{\partial \theta} & \frac{\partial f}{\partial t} \\
0 & 0 & 0 \\
0 & 0 & 0
\end{array}\right]
$$
直接将$f_{aug}$ 和$a_{aug}$ 代入上一小节的伴随法公式：
$$
\frac{d \mathbf{a}_{a u g}(t)}{d t}=-\left[\begin{array}{lllll}
\mathbf{a}(t) & \mathbf{a}_{\theta}(t) & \mathbf{a}_{t}(t)
\end{array}\right] \frac{\partial f_{\text {aug}}}{\partial[\mathbf{z}, \theta, t]}(t)=-\left[\begin{array}{lll}
\mathbf{a} \frac{\partial f}{\partial \mathbf{z}} & \mathbf{a} \frac{\partial f}{\partial \theta} & \mathbf{a} \frac{\partial f}{\partial t}
\end{array}\right](t)
$$
于是得到了最终的结论：
$$
\frac{d L}{d \theta}=\int_{t_{N}}^{t_{0}} \mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \theta} d t
$$

$$
\frac{d L}{d t_{N}}=-\mathbf{a}\left(t_{N}\right) \frac{\partial f\left(\mathbf{z}\left(t_{N}\right), t_{N}, \theta\right)}{\partial t_{N}} \quad \frac{d L}{d t_{0}}=\int_{t_{N}}^{t_{0}} \mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial t} d t
$$

## 梯度变元定理的证明

给定常微分方程：
$$
\frac{dz(t)}{d(t)}=f(z(t),t)
$$
$f$ 要求对$z$ Lipschitz连续，对$t$ 连续。则对数概率密度满足：
$$
\frac{\partial \log p(\mathbf{z}(t))}{\partial t}=-\operatorname{tr}\left(\frac{d f}{d \mathbf{z}}(t)\right)
$$
证明：

首先类似上面伴随法证明的过程，将$z(t+\varepsilon)$ 表示为$T_{\varepsilon}(\mathbf{z}(t))$

$f$ 要求对$z$ Lipschitz连续，对$t$ 连续。这是为了使方程满足Picard存在定理，使得解存在且唯一。

首先是要推导出
$$
\frac{\partial \log p(z(t))}{\partial t}=-\operatorname{tr}\left(\lim _{\varepsilon \rightarrow 0^{+}} \frac{\partial}{\partial \epsilon} \frac{\partial}{\partial z} T_{\varepsilon}(z(t))\right)
$$
过程：
$$
\begin{aligned} \frac{\partial \log p(\mathbf{z}(t))}{\partial t} 
&=\lim _{\varepsilon \rightarrow 0^{+}} \frac{\log p(\mathbf{z}(t))-\log \left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|-\log p(\mathbf{z}(t))}{\varepsilon} \\ &=-\lim _{\varepsilon \rightarrow 0^{+}} \frac{\log \left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|}{\varepsilon}\\
&=-\lim _{\varepsilon \rightarrow 0^{+}} \frac{\frac{\partial}{\partial \varepsilon} \log \left|\operatorname{det} \frac{\partial}{\partial z} T_{\varepsilon}(\mathbf{z}(t))\right|}{\frac{\partial}{\partial \varepsilon} \varepsilon}\\
&=-\lim _{\varepsilon \rightarrow 0^{+}} \frac{\frac{\partial}{\partial \varepsilon}\left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|}{\left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|} \qquad \qquad \quad \left(\left.\frac{\partial \log (\mathbf{z})}{\partial \mathbf{z}}\right|_{\mathbf{z}=1}=1\right)\\
&=-\underbrace{\left(\lim _{\varepsilon \rightarrow 0+} \frac{\partial}{\partial \varepsilon}\left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|\right)}_{\text {bounded }}\left(\lim _{\varepsilon \rightarrow 0^{+}} \frac{1}{\left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|}\right)\\
&=-\lim _{\varepsilon \rightarrow 0^{+}} \frac{\partial}{\partial \varepsilon}\left|\operatorname{det} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right|
\end{aligned}
$$

第一步用到的是流模型中的公式(本质是概率密度上的雅可比公式)，后面仅用到洛必达法则、链式法则等简单技巧。

然后应用雅可比公式：
$$
\begin{aligned}
\frac{\partial \log p(\mathbf{z}(t))}{\partial t} &=-\lim _{\varepsilon \rightarrow 0^{+}} \operatorname{tr}\left(\operatorname{adj}\left(\frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right) \frac{\partial}{\partial \varepsilon} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right) \\
&=-\operatorname{tr}\left(\underbrace{\left(\lim _{\varepsilon \rightarrow 0^{+}} \operatorname{adj}\left(\frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right)\right)}_{=I}\left(\lim _{\varepsilon \rightarrow 0^{+}} \frac{\partial}{\partial \varepsilon} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right)\right) \\
&=-\operatorname{tr}\left(\lim _{\varepsilon \rightarrow 0^{+}} \frac{\partial}{\partial \varepsilon} \frac{\partial}{\partial \mathbf{z}} T_{\varepsilon}(\mathbf{z}(t))\right)
\end{aligned}
$$
雅可比公式是：
$$
\frac{d}{dt}det(A(t))=tr(adj(A(t))\frac{d}{dt}A(t))
$$
最后进行泰勒展开即可：
$$
\begin{aligned}
\frac{\partial \log p(\mathbf{z}(t))}{\partial t} &=-\operatorname{tr}\left(\lim _{\varepsilon \rightarrow 0^{+}} \frac{\partial}{\partial \varepsilon} \frac{\partial}{\partial \mathbf{z}}\left(\mathbf{z}+\varepsilon f(\mathbf{z}(t), t)+\mathcal{O}\left(\varepsilon^{2}\right)+\mathcal{O}\left(\varepsilon^{3}\right)+\ldots\right)\right) \\
&=-\operatorname{tr}\left(\lim _{\varepsilon \rightarrow 0^{+}} \frac{\partial}{\partial \varepsilon}\left(I+\frac{\partial}{\partial \mathbf{z}} \varepsilon f(\mathbf{z}(t), t)+\mathcal{O}\left(\varepsilon^{2}\right)+\mathcal{O}\left(\varepsilon^{3}\right)+\ldots\right)\right) \\
&=-\operatorname{tr}\left(\lim _{\varepsilon \rightarrow 0^{+}}\left(\frac{\partial}{\partial \mathbf{z}} f(\mathbf{z}(t), t)+\mathcal{O}(\varepsilon)+\mathcal{O}\left(\varepsilon^{2}\right)+\ldots\right)\right) \\
&=-\operatorname{tr}\left(\frac{\partial}{\partial \mathbf{z}} f(\mathbf{z}(t), t)\right)
\end{aligned}
$$


