---
date: 2020-02-15 21:19:00
description: VAE、GAN和FLOW_based_model的综述和对比
title: gan,vae和flow
author: 鱼摆摆
comments: true
tags: 
 - gan
 - vae
 - flow
photos: http://img.article.pchome.net/00/32/50/36/pic_lib/wm/ghibli_22.jpg
categories: 学习
---



# 前言

GAN，VAE和FLOW的目标是一致的——希望构建一个从隐变量$Z$生成目标数据$X$的模型，其中先验分布$P(z)$通常被设置为高斯分布。我们希望找到一个变换函数$f(x)$，他能建立一个从$z$到$x$的映射：$f:z\to x$，然后在$P(Z)$中随机采样一个点$z'$，通过映射$f$，就可以找到一个新的样本点$x'$。  

![image.png](https://i.loli.net/2020/02/06/LpCERGiuF4ahBDN.png)

举个栗子：

如何将均匀分布$U[0,1]$映射成正态分布$N(0,1)$？

将$X \sim U[0,1]$经过函数$Y = f(x)$映射之后，就有$Y\sim N(0,1)$了那么$[x,x+dx]$和$[y,y+dy]$两个区间上的概率应该相等，即：

$$
\rho(x) d x=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{y^{2}}{2}\right) d y
$$

对其进行积分，有：
$$
\int_{0}^{x} \rho(t) d t=\int_{-\infty}^{y} \frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{t^{2}}{2}\right) d t=\Phi(y)
$$
$$
y=\Phi^{-1}\left(\int_{0}^{x} \rho(t) d t\right)=f(x)
$$

可以看到$Y = f(X)$的解是存在的，但很复杂，无法用初等函数进行显示的表示，因此在大多数情况下，我们都是通过神经网络来拟合这个函数。

假设我们现在已经有一个映射$f$，我们如何衡量映射$f$构造出来的数据集$f(z_1),f(z_2),...,f(z_n)$，是否和目标数据$X$分布相同？(注：KL和JS距离根据两个概率分布的表达式计算分布的相似度，而我们现在只有从构造的分布采样的数据和真实分布采样的数据，而离散化的KL和JS距离因为图像维度问题，计算量非常大)。在这里GAN采用了一个暴力的办法：训练一个判别器作为两者相似性的度量，而VAE(变分自编码器)和FLOW(流模型)在最大化最大似然。

# VAE(变分自编码器)

## VAE的基本思路

对于连续随机变量，概率分布$P$和$Q$，KL散度(又称相对熵)的定义为：
$$
D_{\mathrm{KL}}(P \| Q)=\int_{-\infty}^{\infty} p(x) \ln \frac{p(x)}{q(x)} \mathrm{d} x=E_{x \sim P(x)}[logP(x)-logQ(x)]
$$
给定一个概率分布$D$,已知其概率密度函数(连续分布)或概率质量函数()离散分布为$f_D$，以及一个分布参数$\theta$，我们可以从这个分布中抽出一个具有$n$个值的采样$X_1,X_2,...,X_n$，利用$f_D$计算出其似然函数：
$$
\mathbf{L}\left(\theta | x_{1}, \ldots, x_{n}\right)=f_{\theta}\left(x_{1}, \ldots, x_{n}\right)
$$


 若$D$是离散分布，$f_{\theta}$即是在参数为$\theta$时观测到这一采样的概率。若其是连续分布，$f_{\theta}$则为$X_1,X_2,...,X_n$联合分布的概率密度函数在观测值处的取值。一旦我们获得$X_1,X_2,...,X_n$，我们就能求得一个关于$\theta$的估计。最大似然估计会寻找关于的最可能的值（即，在所有可能的取值中，寻找一个值使这个采样的“可能性”最大化）。从数学上来说，我们可以在的所有可能取值中寻找一个值使得似然函数取到最大值。这个使可能性最大的值即称为的最大似然估计。由定义，最大似然估计是样本的函数。

VAE做最大似然估计，也就是要最大化概率：
$$
P(X)=\sum_{i} P\left(X | z_{i} ; \theta\right) P\left(z_{i}\right)
$$

这里可以理解为使用积分创造更多的分布，一般选择$P(Z)$服从一个高斯分布，而$p(X|z)$可以是任意分布，例如条件高斯分布或狄拉克分布，理论上讲，这个积分形式的分布可以拟合任意分布。

但是这里的$P(X)$是积分形式的，很难进行计算。VAE从让人望而生畏的变分和贝叶斯理论出发，推导出了一个很接地气的公式：
$$
\log P(X)-\mathcal{D}[Q(z | X) \| P(z | X)]=E_{z \sim Q}[\log P(X | z)]-\mathcal{D}[Q(z | X) \| P(z)] \tag{1}
$$
VAE并没有选择直接去优化$P(X)$，而是选择去优化他的一个变分下界（公式1右端）。

而VAE的自编码器性质也从这个公式里开始体现出来：我们可以将$\mathcal{D}[Q(z | X) \| P(z)]$视作编码器的优化，使由真实数据编码出的隐变量分布$Q(z|X)$去尽量近似$P(z)$（标准高斯分布），而将$E_{z \sim Q}[\log P(X | z)]$视作解码器的优化，使得服从分布$Q$的隐变量$z$解码出的$x$尽可能地服从真是数据分布，而将$\mathcal{D}[Q(z | X) \| P(z | X)]$视作误差项。

但VAE也因为它并没有直接去优化$P(X)$，而选择去优化它的变分下界，使得他只是一个近似模型，无法保证良好的生成效果。

## VAE的优化过程

首先要确定概率密度$Q(z|X)$的形式，一般选择正态分布，即$\mathcal{N}\left(\mu, \sigma^{2}\right)$，其中$\mu\left(X ; \theta_{\mu}\right) , \sigma^{2}\left(X ; \theta_{\sigma}\right)$通过两个神经网络(编码器)训练出来。公式中的$\mathcal{D}[Q(z | X) \| P(z)]$变为$D\left[\mathcal{N}\left(\mu\left(X ; \theta_{\mu}\right), \sigma^{2}\left(X ; \theta_{\sigma}\right)\right) \| \mathcal{N}(0, I)\right]$，这个时候就可以通过两个正态分布的KL散度的计算公式来计算这一项。

对于第一项$E_{z \sim Q}[\log P(X | z)]$，对于一个batch来说，可以在$Q$中采样，然后将单个样本的$\log P(X|z)$求和取平均数作为期望的估计。但这样出现一个问题：把$Q(z|X)$弄丢了，也就是每次训练的时候梯度不传进$Q$里，论文里采用了一个称为重参数化技巧(reparamenterization trick)的方法，如图：

![image.png](https://i.loli.net/2020/02/07/a8ofIEU4XyZdRL2.png)

至此，整个VAE网络就可以训练了。

## 公式推导部分

$$
\begin{aligned} \mathcal{D}[Q(z|X)||P(z|X)] &= E_{z \sim Q}[\log Q(z|X) - \log P(z|X)] \\ &= E_{z \sim Q}[\log Q(z|X) - \log P(z|X) - \log P(X)] + \log P(X) \end{aligned}
$$

移项得
$$
\log P(X)-\mathcal{D}[Q(z | X) \| P(z | X)]=E_{z \sim Q}[\log P(X | z)]-\mathcal{D}[Q(z | X) \| P(z)]
$$

# GAN

## 模型构建

由于大家都对GAN比较熟悉，本文直接从变分推断的角度去理解GAN。

不同于VAE将$P(X|z)$选为高斯分布，GAN的选择是：
$$
P(x | z)=\delta(x-G(z)), \quad P(x)=\int P(x | z) P(z) d z
$$
其中$\delta (x)$是狄拉克函数，$G(z)$为生成器网络。

在VAE中z被当作是一个隐变量，但在GAN中，狄拉克函数意味着单点分布，即x和z为一一对应的关系。于是在GAN中z没有被当作隐变量处理(不需要考虑后验分布$P(z|x)$)

判别器的理解：

在GAN中引入了一个二元的隐变量y来构成联合分布，其中$\tilde{p}(x)$ 为真实样本的分布：
$$
q(x, y)=\left\{\begin{array}{l}
{\tilde{p}(x) p_{1}, y=1} \\
{p(x) p_{0}, y=0}
\end{array}\right.
$$
这里y是图像的真实标签，当图片为真实图片时，y=1，当图片是生成图片时，y=0。

其中$p_1+p_0=1$描述了一个二元概率分布，比如：从真实样本采集m个样本，从生成样本中采集m个样本，同时传入判别器，则$p_0=p_1=1/2$。在下面讨论中我们直接取$p_0=p_1=1/2$

另一方面，我们需要使判别器的判别结果尽可能真实，设$p(x,y)=p(y|x)\tilde{p}(x)$，$p(y|x)$为一个条件伯努利分布(判别器的判别结果)。优化目标是$KL(q(x,y)||p(x,y))$：
$$
\begin{aligned}
K L(q(x, y) \| p(x, y)) &=\int \tilde{p}(x) p_{1} \log \frac{\tilde{p}(x) p_{1}}{p(1 | x) \tilde{p}(x)} d x+\int p(x) p_{0} \log \frac{p(x) p_{0}}{p(0 | x) \tilde{p}(x)} d x \\
& \sim \int \tilde{p}(x) \log \frac{1}{2p(1 | x)} d x+\int p(x) \log \frac{p(x)}{2p(0 | x) \tilde{p}(x)} d x\\
& = -E_{x \sim \tilde{p}(x)}[\log 2p(1|x)]-E_{x \sim p(x)}[\log 2p(0|x)]+KL(p(x)||\tilde{p}(x))
\end{aligned}
$$
一旦成功优化，就有$q(x,y)\to p(x,y)$，对于x求边缘概率分布，有：
$$
\frac{1}{2}\tilde{p}(x)+\frac{1}{2}p(x)\to p(1|x)\tilde{p}(x)+p(0|x)\tilde{p}(x)=\tilde{p}(x)
$$
即：
$$
p(x)\to \tilde{p}(x)
$$
这就完成了对模型的构建。

## 目标优化

现在我们有优化目标：$p(1|x)$和$G(z)$，分别是判别器($p(y|x)$服从条件伯努利分布，可以直接由$p(1|x)$确定)和生成器($p(x)$由$G(z)$决定)。类似EM算法，我们进行交替优化：先固定$G(z)$,这也意味着$p(x)$固定了，然后优化$p(y|x)$，优化目标为：
$$
D=\underset{D}{\arg \min }\{-E_{x \sim \tilde{p}(x)}[\log 2D(x)]-\mathbb{E}_{x \sim p(x)}[\log 2(1-D(x))]\}
$$
然后固定$D(x)$来优化$G(x)$，相关loss为：
$$
G=\underset{G}{\arg \min } \int p(x) \log \frac{p_0 p(x)}{(1-D(x)) \tilde{p}(x)} d x
$$
假设$D(x)$有足够的拟合能力，注意到当$D(x)=\frac{\tilde{p}(x)}{\tilde{p}(x)+p^0(x)}$时，有
$$
\begin{aligned}
K L(q(x, y) \| p^0(x, y)) &= \int \tilde{p}(x) \log \frac{1}{2D(x)} d x+\int p^0(x) \log \frac{p^0(x)}{2(1-D(x)) \tilde{p}(x)} d x\\
&= \int\tilde{p}(x) \log \frac{\tilde{p}(x)+p^0(x)}{2\tilde{p}(x)}+p^0(x) \log \frac{\tilde{p}(x)+p^0(x)}{2\tilde{p}(x)}\\
&= KL(\tilde{p}(x) +p^0(x)||2\tilde{p}(x) )
\end{aligned}
$$
不严格的说法：由于现在对$p^0(x)$ 和 $\tilde{p}(x)$没有约束，可以直接$p^0(x)=\tilde{p}(x)$使得loss等于0，也就是说$D(x)=\frac{\tilde{p}(x)}{\tilde{p}(x)+p^0(x)}$为理论最优解。在优化判别器时，$p^0(x)$应该为上一阶段生成器优化的$p(x)$ 。将这个$D(x)$代入生成器的相关loss：
$$
\begin{aligned}
G &= \underset{G}{\arg \min } \int p(x) \log \frac{p_0 p(x)}{(1-D(x)) \tilde{p}(x)} d x\\
&= \underset{G}{\arg \min } \int p(x) \log \frac{ p(x)}{2D(x) p^0(x)} d x\\
&= \underset{G}{\arg \min }[-E_{x \sim p(x)}2D(x)+KL(p(x)||p^0(x))]\\
&= \underset{G}{\arg \min }[-E_{x \sim p(x)}2D(G(z))+KL(p(x)||p^0(x))]
\end{aligned}
$$
可以看到，此时的第一项$-E_{x \sim p(x)}2D(G(z))$就是标准的GAN所采用的loss之一。而我们知道，目前标准的GAN生成器的loss都不包含$KL(p(x)||p^0(x))$，这实际上造成了loss的不完备。

顺便提一句，VAE中也有类似GAN中交替优化的方法，称为EM算法。

第二个loss是在限制要求新的生成器跟旧的生成器生成结果不能差别太大 ，也就是生成器不能剧烈变化。在loss不完备的情况下，假设有一个优化算法总能找到$G(z)$的理论最优解、并且$G(z)$具有无限的拟合能力，那么$G(z)$只需要生成唯一一个使得$D(x)$最大的样本（不管输入的$z$是什么），这就是模型坍缩。模型塌缩的[视频](https://www.youtube.com/watch?v=Co2ukCewKkE )(需要梯子)。

然后对第二项进行估算，得到一个可以在实验中使用的正则项：

记$p^{o}(x)=q_{\theta-\Delta \theta}(x), \quad p(x)=q_{\theta}(x)$，其中$\Delta \theta$为生成器的参数变化，对$q^{o}(x)=q_{\theta-\Delta \theta}(x)$做泰勒展开，有：
$$
q^{o}(x)=q_{\theta-\Delta \theta}(x)=q_{\theta}(x)-\Delta \theta \cdot \nabla_{\theta} q_{\theta}(x)+O\left((\Delta \theta)^{2}\right)
$$

$$
\begin{aligned}
K L\left(q(x) \| q^{o}(x)\right) & \approx \int q_{\theta}(x) \log \frac{q_{\theta}(x)}{q_{\theta}(x)-\Delta \theta \cdot \nabla_{\theta} q_{\theta}(x)} d x \\
&=-\int q_{\theta}(x) \log \left[1-\frac{\Delta \theta \cdot \nabla_{\theta} q_{\theta}(x)}{q_{\theta}(x)}\right] d x \\
& \approx-\int q_{\theta}(x)\left[-\frac{\Delta \theta \cdot \nabla_{\theta} q_{\theta}(x)}{q_{\theta}(x)}-\left(\frac{\Delta \theta \cdot \nabla_{\theta} q_{\theta}(x)}{q_{\theta}(x)}\right)^{2}\right] d x \\
&=\Delta \theta \cdot \nabla_{\theta} \int q_{\theta}(x) d x+(\Delta \theta)^{2} \cdot \int \frac{\left(\nabla_{\theta} q_{\theta}(x)\right)^{2}}{2 q_{\theta}(x)} d x \\
&=(\Delta \theta)^{2} \cdot \int \frac{\left(\nabla_{\theta} q_{\theta}(x)\right)^{2}}{2 q_{\theta}(x)} d x \\
& \approx(\Delta \theta \cdot c)^{2}
\end{aligned}
$$

上式中应用了$\log(1+x)$的泰勒展开式以及求导和积分可互换、可积分的假设。上面的粗略估计表明，生成器的参数不能变化太大。而我们用的是基于梯度下降的优化算法，所以$\Delta \theta$正比于梯度，因此标准GAN训练时的很多trick，比如梯度裁剪、用Adam优化器、用BN，都可以解释得通了，它们都是为了稳定梯度，使得$\Delta \theta$不至于过大，同时，$G(z)$的迭代次数也不能过多，因为过多同样会导致$\Delta \theta$过大。 

## 正则项

考虑如何添加正则项以改进GAN的稳定性：

直接对$K L\left(q(x) \| q^{o}(x)\right)$进行估算是很困难的，但是我们上面提到$q(z|x)$和$q^o(z|x)$是狄拉克分布，而狄拉克分布可以看作方差为0的高斯分布，于是考虑用$K L\left(q(x,z) \| q^{o}(x,z)\right)$进行估算：
$$
\begin{aligned}
K L(q(x, z) \| \tilde{q}(x, z)) &=\iint q(x | z) q(z) \log \frac{q(x | z) q(z)}{\tilde{q}(x | z) q(z)} d x d z \\
&=\iint \delta(x-G(z)) q(z) \log \frac{\delta(x-G(z))}{\delta\left(x-G^{o}(z)\right)} d x d z \\
&=\int q(z) \log \frac{\delta(0)}{\delta\left(G(z)-G^{o}(z)\right)} d z
\end{aligned}
$$
将狄拉克分布可以看作方差为0的高斯分布,并代入：
$$
\delta(x)=\lim _{\sigma \rightarrow 0} \frac{1}{\left(2 \pi \sigma^{2}\right)^{d / 2}} \exp \left(-\frac{x^{2}}{2 \sigma^{2}}\right)
$$

$$
\begin{aligned}
K L(q(x, z) \| \tilde{q}(x, z)) &=\int q(z) \log \frac{\delta(0)}{\delta\left(G(z)-G^{o}(z)\right)} d z \\
&= \lim _{\sigma \rightarrow 0} \int q(x) \log \left[ 1/{\exp \left(-\frac{(G(z)-G^0(z))^{2}}{2 \sigma^{2}}\right)} \right]dx \\
&= \lim _{\sigma \rightarrow 0} \int q(x)  \left(-\frac{(G(z)-G^0(z))^{2}}{2 \sigma^{2}}\right)dx \\
& \sim \lambda \int q(z)\left\|G(z)-G^{o}(z)\right\|^{2} d z
\end{aligned}
$$

于是有
$$
K L\left(q(x) \| q^{o}(x)\right) \sim \lambda \int q(z)\left\|G(z)-G^{o}(z)\right\|^{2} d z
$$
从而完整的生成器loss可以选择为
$$
\mathbb{E}_{z \sim q(z)}\left[-\log D(G(z))+\lambda\left\|G(z)-G^{o}(z)\right\|^{2}\right]
$$

## 实验结果

![1581163282072.png](https://i.loli.net/2020/02/09/qNQYlSrw5mAoKWM.png)

![1581163339120.png](https://i.loli.net/2020/02/09/czJFlsryfVOSjn6.png)

# FLOW

基本思路：直接硬算积分式
$$
\int_{z} p(x | z) p(z) d z
$$

流模型有一个非常与众不同的特点是，它的转换通常是可逆的。也就是说，流模型不 仅能找到从 A 分布变化到 B 分布的网络通路，并且该通路也能让 B 变化到 A，简言之流模 型找到的是一条 A、B 分布间的双工通路。当然，这样的可逆性是具有代价的——A、B 的 数据维度必须是一致的。

 A、B 分布间的转换并不是轻易能做到的，流模型为实现这一点经历了三个步骤：最初 的 NICE 实现了从 A 分布到高斯分布的可逆求解；后来 RealNVP 实现了从 A 分布到条件非 高斯分布的可逆求解；而最新的 GLOW，实现了从 A 分布到 B 分布的可逆求解，其中 B 分 布可以是与 A 分布同样复杂的分布，这意味着给定两堆图片，GLOW 能够实现这两堆图片 间的任意转换。 


## NICE

两个一维分布之间的转化参考前言中的栗子，下面考虑高维分布：

![image.png](https://i.loli.net/2020/02/10/7IvQ1elFWSjgAOz.png)

类似一维分布，两个分布在映射前后的相同区域应该有相同的概率。
$$
p\left(x^{\prime}\right)\left|\operatorname{det}\left(J_{f}\right)\right|=\pi\left(z^{\prime}\right)
$$
其中$J_f$为雅可比行列式，函数$f$将$z$上的分布变换到$x$上的分布。

根据雅可比行列式的逆运算，同样有：
$$
p\left(x^{\prime}\right)=\pi\left(z^{\prime}\right)\left|\operatorname{det}\left(J_{f^{-1}}\right)\right|
$$
至此，我们得到了一个比较重要的结论：如果 $z$ 与 $x$ 分别满足两种分布，并且 $z$ 通过 函数 $f$ 能够转变为 $x$，那么 $z$ 与 $x$ 中的任意一组对应采样点 $𝑧′$ 与 $𝑥′$ 之间的关系为： 
$$
\left\{\begin{array}{c}
{\pi\left(z^{\prime}\right)=p\left(x^{\prime}\right)\left|\operatorname{det}\left(J_{f}\right)\right|} \\
{p\left(x^{\prime}\right)=\pi\left(z^{\prime}\right)\left|\operatorname{det}\left(J_{f^{-1}}\right)\right|}
\end{array}\right.
$$
从这个公式引入了Flow_based_model 的基本思路：设计一个神经网络，将分布 $x$ 映射到分布 $z$ ，具体来说，流模型选择 $q(z)$ 为高斯分布，$q(x|z)$ 为狄拉克分布 $\delta(x-g(z)$ ，其中$g$ 是可逆的：
$$
x=g(z) \Leftrightarrow z=f(x)
$$
要从理论上实现可逆，需要 $x$ 和 $z$ 的维数相同，将 $z$ 的分布代入，则有：
$$
q(z)=\frac{1}{(2 \pi)^{D / 2}} \exp \left(-\frac{1}{2}\|z\|^{2}\right)
$$

$$
q(\boldsymbol{x})=\frac{1}{(2 \pi)^{D / 2}} \exp \left(-\frac{1}{2}\|\boldsymbol{f}(\boldsymbol{x})\|^{2}\right)\left|\operatorname{det}\left[\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\right]\right|\tag{2}
$$

公式$(2)$对 $f$ 提出了三个基本要求： 

- 可逆，且逆函数容易计算。
- 对应的雅可比行列式方便计算
- 拟合能力强

这样的话，就有
$$
\log q(x)=-\frac{D}{2} \log (2 \pi)-\frac{1}{2}\|f(x)\|^{2}+\log \left|\operatorname{det}\left[\frac{\partial f}{\partial x}\right]\right|
$$
这个优化目标是可计算的，并且因为 $f$ 可逆，那么我们在$z$ 中取样，就可以生成相应的 $x$

$$
x=f^{-1}(z)=g(z)
$$
为了满足这三个条件，NICE和REAL NVP、GLOW都采用了模块化思想，将 $f$ 设计成一组函数的复合，其中每个函数都满足要求一和要求二，经过复合之后函数也容易满足要求三。
$$
f=f_{L} \circ \ldots \circ f_{2} \circ f_{1}
$$
相对而言，雅可比行列式的计算要比函数求逆更加复杂，考虑第二个要求，我们知道三角行列式最容易计算，所以我们要想办法让变换 $f$ 的雅可比矩阵为三角阵。NICE的做法是：将 $D$ 的 $x$ 分为两部分 $x_1,x_2$，然后取下述变换：
$$
\begin{array}{l}
{\boldsymbol{h}_{1}=\boldsymbol{x}_{1}} \\
{\boldsymbol{h}_{2}=\boldsymbol{x}_{2}+\boldsymbol{m}\left(\boldsymbol{x}_{1}\right)}
\end{array}
$$
其中 $m$ 为任意函数，这个变换称为“加性耦合层” ，这个变换的雅可比矩阵 $[\frac{\partial h}{\partial x}]$ 是一个三角阵，且对角线元素全部为1，用分块矩阵表示为：
$$
\left[\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}\right]=\left(\begin{array}{cc}
{\mathrm{I}_{d}} & {\mathrm{O}} \\
\frac{\partial m}{\partial x_1} & I_{D-d}
\end{array}\right)
$$
同时这个变换也是可逆的，其逆变换为
$$
\begin{array}{l}
{\boldsymbol{x}_{1}=\boldsymbol{x}_{h}} \\
{\boldsymbol{x}_{2}=\boldsymbol{h}_{2}-\boldsymbol{m}\left(\boldsymbol{h}_{1}\right)}
\end{array}
$$

满足了要求一和要求二，同时这个雅可比行列式的值为1，行列式的值的物理含义是体积，所以这个变换暗含了变换前后的体积不变性。我们注意到：该变换的第一部分是平凡的（恒等变换），因此需要对调I1和I2两组维度，再输入加和耦合层，并将这个过程重复若干次， 以达到信息充分混合的目的，如图：

![image.png](https://i.loli.net/2020/02/10/n9AHawOlYcM5PiC.png) 

因为该变换需要满足 $z$ 和 $x$ 的维度相同，这会产生很严重的唯独浪费问题，NICE在最后一层里引入了一个尺度变换对维度进行缩放：
$$
z=s \otimes h^{(n)}
$$
其中$s=(s_1,s_2,...,s_D)$也是一个要优化的参数向量，这个 $s$ 向量能够识别每个维度的重要程度， $s$ 越小，这个维度越不重要，起到压缩流形的作用。这个尺度变换层的雅可比行列式就不是一了，而是：
$$
\left[\frac{\partial z}{\partial \boldsymbol{h}^{(n)}}\right]=\operatorname{diag}(\boldsymbol{s})
$$
他的行列式的值为 $\prod_{i} s_{i}$,于是最后的对数似然为：
$$
\log q(\boldsymbol{x}) \sim-\frac{1}{2}\|\boldsymbol{s} \otimes \boldsymbol{f}(\boldsymbol{x})\|^{2}+\sum_{i} \log \boldsymbol{s}_{i}
$$
这个尺度变换实际上是将先验分布 $q(z)$ 的方差也作为训练参数，方差越小，说明这个维度的“弥散”越小，若方差为0，这一维的特征就恒为均值，于是流行减小一维。

我们写出带方差的正态分布：
$$
q(z)=\frac{1}{(2 \pi)^{D / 2} \prod_{i=1}^{D} \sigma_{i}} \exp \left(-\frac{1}{2} \sum_{i=1}^{D} \frac{z_{i}^{2}}{\sigma_{i}^{2}}\right)
$$
将 $z=f(x)$ 代入，并取对数，类似得：
$$
\log q(\boldsymbol{x}) \sim-\frac{1}{2} \sum_{i=1}^{D} \frac{\boldsymbol{f}_{i}^{2}(\boldsymbol{x})}{\boldsymbol{\sigma}_{i}^{2}}-\sum_{i=1}^{D} \log \boldsymbol{\sigma}_{i}
$$
与之前那个公式对比，就有 $s_i=1/\sigma_i$ ，所以尺度变换层等价于将先验分布的方差作为训练参数，若方差足够小，则维度减一，暗含了降维的可能。

## REALNVP

NICE构思巧妙，但在实验部分只是采取了简单的加性耦合层和将全连接层进行简单的堆叠，并没有使用卷积。REALNVP一般化了耦合曾，并在耦合模型中引入了卷积层，使得模型可以更好地处理图像问题。论文里还引入了一个多尺度结构来处理维度浪费问题。

将加性耦合层换成仿射耦合层：
$$
\begin{array}{l}
{\boldsymbol{h}_{1}=\boldsymbol{x}_{1}} \\
{\boldsymbol{h}_{2}=\boldsymbol{s}\left(\boldsymbol{x}_{1}\right) \otimes \boldsymbol{x}_{2}+t\left(\boldsymbol{x}_{1}\right)\left(\boldsymbol{x}_{1}\right)}
\end{array}
$$
仿射耦合层的雅可比行列式仍然是一个对角阵
$$
\left[\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}\right]=\left(\begin{array}{cc}
{\mathbb{I}_{d}} & {\mathbb{O}} \\
{\left[\frac{\partial m}{\partial x_{1}}\right]} & {s}
\end{array}\right)
$$
雅可比行列式的值不再是1，没有保持变换前后的体积不变。

在NICE中，通过交错的方式来混合信息流(直接反转原来的向量)，在REALNVP中发现：随机打乱维度可以使信息混合的更加充分。

![image.png](https://i.loli.net/2020/02/10/bF8tI2AJgEnd5BQ.png)

![image.png](https://i.loli.net/2020/02/10/3Xm8YzISUqh5irt.png)

引入卷积层：使用卷积的条件是具有局部相关性，因此指定向量的打乱和重排都是在channel维度上进行，在height和width维度上进行卷积。对通道的分割论文里还提出棋盘式分割的策略，但较为复杂，对模型的提升也不大，因此在GLOW中被舍弃了。

一般的图像通道数只有三层，MNIST等灰度图只有一层，因此REALNVP引入了squeeze操作来增加通道数。

其思想很简单：直接 reshape，但 reshape 时局部地进行。具体来说，假设原来图像为 h×w×c 大小，前两个轴是空间维度，然后沿着空间维度分为一个个 2×2×c 的块（ 2 可以自定义），然后将每个块直接 reshape 为 1×1×4c，最后变成了 h/2×w/2×4c。 

![image.png](https://i.loli.net/2020/02/10/bQUZA2O5x73Pghm.png)

REALNVP中还引入了一个多尺度结构：

![image.png](https://i.loli.net/2020/02/10/MfAsK1BCbGJtZvy.png)

最终的输出 $z_1,z_3,z_5$ 怎么取？
$$
p\left(z_{1}, z_{3}, z_{5}\right)=p\left(z_{1} | z_{3}, z_{5}\right) p\left(z_{3} | z_{5}\right) p\left(z_{5}\right)
$$
由于 $z_3,z_5$ 是由 $z_2$ 完全决定的，$z_5$ 也是由 $z_4$ 完全决定的，因此条件部分可以改为： 
$$
p\left(z_{1}, z_{3}, z_{5}\right)=p\left(z_{1} | z_{2}\right) p\left(z_{3} | z_{4}\right) p\left(z_{5}\right)
$$
RealNVP 和 Glow 假设右端三个概率分布都是正态分布，类似VAE， $p(z_1|z_2)$ 的均值方差由 $z_2$ 算出来，$p(z_3|z_4)$  的均值方差由 $z_4$ 算出来，$p(z_5)$  的均值方差直接学习出来。这相当于做了变量代换：
$$
\hat{z}_{1}=\frac{z_{1}-\mu\left(z_{2}\right)}{\sigma\left(z_{2}\right)}, \quad \hat{z}_{3}=\frac{z_{3}-\mu\left(z_{4}\right)}{\sigma\left(z_{4}\right)}, \quad \hat{z}_{5}=\frac{z_{5}-\mu}{\sigma}
$$
然后认为 $[\hat{z}_1,\hat{z}_3,\hat{z}_5]$服从标准正态分布。类似NICE，这三个变换会导致一个非1的雅可比行列式，也就是往loss中加入 $\Sigma_{i=1}^{D} \log \sigma_{i}$ 这一项。

**多尺度结构相当于抛弃了 $p(z)$ 是标准正态分布的直接假设，而采用了一个组合式的条件分布**，这样尽管输入输出的总维度依然一样，但是不同层次的输出地位已经不对等了，模型可以通过控制每个条件分布的方差来抑制维度浪费问题（极端情况下，方差为 0，那么高斯分布坍缩为狄拉克分布，维度就降低 1），条件分布相比于独立分布具有更大的灵活性。而如果单纯从 loss 的角度看，多尺度结构为模型提供了一个强有力的正则项。

## GLOW

效果好的令人惊叹的生成模型：

[改变图像属性](https://v.qq.com/x/page/l30187h002z.html )

[采样展示](https://v.qq.com/x/page/a13440frros.html )

[潜在空间的插值](https://v.qq.com/x/page/m13446r8u6u.html )

总体来说，GLOW引入1*1可逆卷积来代替通道维度的打乱和重排操作，并对REALNVP的原始模型做了简化和规范。

向量之间的元素置换操作可以用简单的行变换矩阵来操作：
$$
\left(\begin{array}{l}
{b} \\
{a} \\
{d} \\
{c}
\end{array}\right)=\left(\begin{array}{llll}
{0} & {1} & {0} & {0} \\
{1} & {0} & {0} & {0} \\
{0} & {0} & {0} & {1} \\
{0} & {0} & {1} & {0}
\end{array}\right)\left(\begin{array}{l}
{a} \\
{b} \\
{c} \\
{d}
\end{array}\right)
$$
GLOW中用一个更一般的矩阵 $W$ 来代替这个置换矩阵
$$
h=xW
$$
这个变换的雅可比矩阵就是$det(W)$，因此需要将 $-log|det(W)|$ 加入到loss中，$W$ 的初始选择要求可逆，不引入loss，因此选为随即正交阵。

这个变换引入了 $det(W)$ 的计算问题，GLOW中逆用LU分解克服了这个问题，若 $W=PLU$ (其中P是一个置换矩阵),则
$$
\log |\operatorname{det} W|=\sum \log |\operatorname{diag}(U)|
$$
这就是GLOW中给出的技巧：先随机生成一个正交矩阵，然后做 $LU$ 分解，得到 $P,L,U$，固定 P，也固定 U 的对角线的正负号，然后约束 L 为对角线全 1 的下三角阵，U 为上三角阵，优化训练 L,U 的其余参数。 

整个GLOW模型如下：

![image.png](https://i.loli.net/2020/02/10/3aK6vM5O2YghIdA.png)

## 对比

比较反转、打乱和1*1逆卷积的loss：

![image.png](https://i.loli.net/2020/02/10/k9MElbZDWc8uHd6.png)



## 缺点

模型庞大，参数量极大，NICE模型在MNIST数据集上的训练参数就大概有两千万个。

再贴两个Glow模型在~~Gayhub~~ Github上的issue感受下：

![image.png](https://i.loli.net/2020/02/10/vmAOIzFpZlyRUiT.png)

256*256的高清人脸生成，用一块GPU训练的话，大概要一年……




# 一图对比GAN，VAE和FLOW

![1581163587153](https://i.loli.net/2020/02/09/c2PZjseRgalKBWI.png)

# 参考文献

- [Variational Inference: A Unified Framework of Generative Models and Some Revelations](https://link.zhihu.com/?target=https%3A//www.paperweekly.site/papers/2117) 
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [用变分推断统一理解生成模型（VAE、GAN、AAE、ALI）](https://kexue.fm/archives/5716)
- [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)
- [NOTE_FLOW](http://www.seeprettyface.com/pdf/Note_Flow.pdf )
- [Glow: Generative Flow with Invertible 1×1 Convolutions](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf )
- [细水长flow之NICE：流模型的基本概念与实现](http://www.sohu.com/a/246846378_500659)
- [RealNVP与Glow：流模型的传承与升华](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/82112222)
- [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803 )

