---
date: 2020-04-24 11:38:00
description: numpy用法介绍
title: Numpy
author: 鱼摆摆
comments: true
tags: 
 - python
 - numpy
photos: https://w.wallhaven.cc/full/6k/wallhaven-6k3oox.jpg
categories: 学习
---

# 基础篇

## 创建数组：

```python
# 通过list创建
>>>np.array([[1, 2], [3, 4]])
array([[1, 2],
       [3, 4]])
# 通过arange创建
>>>np.arange(0,1,0.1)
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# arange+广播
>>>np.arange(1,60,10).reshape(-1,1)+np.arange(0,6)
array([[ 1,  2,  3,  4,  5,  6],
       [11, 12, 13, 14, 15, 16],
       [21, 22, 23, 24, 25, 26],
       [31, 32, 33, 34, 35, 36],
       [41, 42, 43, 44, 45, 46],
       [51, 52, 53, 54, 55, 56]])
# linspace通过等差数列创建数组
>>>np.linspace(0,1,10)
array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
       0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
# 特殊形式数组
>>>np.zeros((2,3),np.int)
array([[0, 0, 0],
       [0, 0, 0]])
>>>np.ones((2,3),np.int)
array([[1, 1, 1],
       [1, 1, 1]])
# 长度为10，元素值为0-1的随机数数组
>>>np.random.rand(2,3)
array([[0.96064533, 0.55490284, 0.13219661],
       [0.3036712 , 0.95073354, 0.39364538]])
# 通过frombuffer,fromstring,fromfile和fromfunction等函数创建数组
>>>np.fromfunction(lambda x,y:(x+1)*(y+1),(2,3))
array([[1., 2., 3.],
       [2., 4., 6.]])
# a和b之间的随机整数
>>>np.random.randint(low=0,high=20,size=5)
array([ 7, 19, 12, 18, 12])

```
## 索引和切片

```python
>>>a=np.arange(5)
>>>a[2]
2
>>>a[:2]
array([0, 1])
>>>a[:-1]
array([0, 1, 2, 3])
# 加入步长
>>>a[0:4:2]
array([0, 2])
>>>a[::-1]
array([4, 3, 2, 1, 0])
# 布尔索引
>>>mask=np.array([True,True,False,False,True])
>>>a[mask]
array([0, 1, 4])
>>>a[a>2]
array([3, 4])
# 索引轴缺失
>>>a=np.arange(6).reshape(2,3)
>>>a[-1]
array([3, 4, 5])
>>>a[-1,:]
array([3, 4, 5])
# 使用...补全索引轴
>>>a=np.arange(24).reshape(2,3,4)
>>>a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>>a[0,...,1]
array([1, 5, 9])
```

## 转置及reshape

**reshape:**

```python
# None可将一维行向量转置为列向量
>>>a=np.random.rand(6)
>>>a
array([0.25779506, 0.22348449, 0.19464385, 0.26307378, 0.76859958,
       0.80357972])
>>>a[:,None]
array([[0.25779506],
       [0.22348449],
       [0.19464385],
       [0.26307378],
       [0.76859958],
       [0.80357972]])
# 等价于reshape
>>>a.reshape(-1,1)
array([[0.25779506],
       [0.22348449],
       [0.19464385],
       [0.26307378],
       [0.76859958],
       [0.80357972]])
>>>a.reshape(2,3)
array([[0.25779506, 0.22348449, 0.19464385],
       [0.26307378, 0.76859958, 0.80357972]])
# flatten()返回一维数组
>>>a=np.arange(6).reshape(2,3)
>>>a.flatten()
array([0, 1, 2, 3, 4, 5])
# 等价于ravel或reshape(-1)
>>>np.ravel(a)
array([0, 1, 2, 3, 4, 5])
```

**transpose:**

```python
>>>a=np.arange(24).reshape(2,3,4)
>>>a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>>a,shape
(2, 3, 4)
>>>a.transpose(2,0,1)
array([[[ 0,  4,  8],
        [12, 16, 20]],
       [[ 1,  5,  9],
        [13, 17, 21]],
       [[ 2,  6, 10],
        [14, 18, 22]],
       [[ 3,  7, 11],
        [15, 19, 23]]])
>>>a.transpose(2,0,1).shape
(4, 2, 3)
```

# Ufun

numpy数学函数：

注：不提供axis时，按整个数组计算

| 函数       | 说明 |
| --------- | ---- |
| np.sin(x) | sin(x) |
| np.cos(x) | cos(x) |
| np.tan(x | tan(x) |
| np.arcsin(x) | arcsin(x) |
|np.arccos(x) | arccos(x) |
|np.arctan(x)| arctan(x) |
|np.arctan2(x,y)| arctan(x/y) |
|np.deg2rad(x)| 角度转弧度 |
|np.rad2deg(x)| 弧度转角度 |
|np.prod(x,axis=None)| 乘积 |
|np.sum(x,axis=None)| 求和 |
|np.exp(x)| exp(x) |
|np.log(x)| ln(x) |
|np.sqrt(x)| 开根 |
|np.square(x)| 平方 |
|np.absolute(x)| 绝对值 |
|np.fabs(x)| 绝对值 |
|np.sign(x)| 符号 |
|np.maximum(x,y)| 逐元素取最大值 |
|np.minimum(x,y)| 逐元素取最小值 |
| np.mean(x,axis=None) | 均值 |
| np.std(x,axis=None) | 标准差 |
| np.var(x,axis=None) | 方差 |
| np.average(x,weight) | (加权)平均 |
| np.argmax(x,axis=None) | 最大值索引 |
| np.argmin(x,axis=None) | 最小值索引 |
| np.sort(x,axis=None) | 从小到大排序 |
| np.argsort(x,axis=None) | 从小到大排序的索引 |

**利用frompyfunc自定义Ufun：**

`np.frompyfunc(func,n_in,n_out)`

```python
>>>def pow2(x):
...    return x**2
>>>fun_pow2=np.frompyfunc(pow2,1,1)
>>>a=np.arange(5)
>>>fun_pow2(a)
array([0, 1, 4, 9, 16], dtype=object)
```

# 广播操作

>广播是针对形状不同的数组的运算采取的操作。
>当我们使用ufunc函数对两个数组进行计算时，ufunc函数会对这两个数组的对应元素进行计算，因此它要求这两个数组有相同的大小(shape相同)。如果两个数组的shape不同的话（行列规模不等），会进行如下的广播(broadcasting)处理：
>1）. 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐。因此输出数组的shape是输入数组shape的各个轴上的最大值（往最大轴长上靠）。
>2）. 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错。
>3）. 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值。

```python
>>>np.arange(2)[:,None]+np.arange(3)
array([[0, 1, 2],
       [1, 2, 3]])
```

# 四则运算

**+，-，*，/ ，**  为逐元素四则运算**

**矩阵乘法，注意要符合矩阵乘法规则:**

```python
# 2*3矩阵
>>>a=np.array([[1,2,3],[4,5,6]])
# 3*2矩阵
>>>b=np.array([[3,4],[5,6],[7,8]])
>>>a.dot(b)
array([[34, 40],
       [79, 94]])
```

**内积：**对于两个一维数组，计算的是这两个数组对应下标元素的乘积和；对于多维数组a和b，它计算的结果数组中的每个元素都是数组a和b的最后一维的内积，因此数组a和b的最后一维的长度必须相同。

计算公式为：**inner(a, b)[i,j,k,m] = sum(a[i,j,:]\*b[k,m,:])**

```python
>>>a=np.arange(12).reshape(2,3,2)
>>>b=np.arange(12,24).reshape(2,3,2)
>>>np.inner(a,b)
array([[[[ 13,  15,  17],
         [ 19,  21,  23]],
        [[ 63,  73,  83],
         [ 93, 103, 113]],
        [[113, 131, 149],
         [167, 185, 203]]],
       [[[163, 189, 215],
         [241, 267, 293]],
        [[213, 247, 281],
         [315, 349, 383]],
        [[263, 305, 347],
         [389, 431, 473]]]])
```

**外积：**只按照一维数组进行计算，如果传入为多维数组，先展开再计算

```python
>>>np.outer([1,2,3],[4,5,6,7])
array([[ 4,  5,  6,  7],
       [ 8, 10, 12, 14],
       [12, 15, 18, 21]])
```



# 其它一些

**np.ndenumerate**返回索引及数组值的迭代对象

```python
>>>for index, x in np.ndenumerate(c):
...    print(index, x)
((0, 0), 1)
((0, 1), 2)
((1, 0), 3)
((1, 1), 4)

>>>np.ndenumerate(c)
<numpy.lib.index_tricks.ndenumerate at 0x7f21cc0dbb90>

>>>np.ndenumerate(c).next()
((0, 0), 1)
```



**np.random.choice**从一维数组或int对象随机选择元素

np.random.choice(a,size=None,replace=True,p=None)

a:一维数据或int对象；replace=True：可重复选择；p：选取的概率

```python
>>>np.random.choice(5,3,p=[0.1,0,0.3,0.6,0])
Out[4]: array([3, 3, 2], dtype=int64)
```



**np.nonezeros**返回非0元素的索引

```python
>>>np.nonzero([[0,1,2],[0,0,2]])
(array([0, 0, 1], dtype=int64), array([1, 2, 2], dtype=int64))
```



**np.intersect1d**求两个数组的交集：

```python
>>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1]) 
array([1, 3])
# 利用reduce取多个交
>>> from functools import reduce
>>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
array([3])
```



**np.where**找到矩阵中满足条件的元素的索引

```python
>>> x = np.arange(9.).reshape(3, 3)
>>> np.where( x > 5 )
(array([2, 2, 2]), array([0, 1, 2]))
>>> x[np.where( x > 3.0 )]               # 返回大于3的值.
array([ 4.,  5.,  6.,  7.,  8.])
>>> np.where( x == 3.0 )             # 返回等于3的值的索引.
(array([1], dtype=int64), array([0], dtype=int64))
```



**np.indices:**获取数组shape属性的所有索引，其shpe为(dim,shape)

```python
>>>np.indices((2,3)).shape
(2, 2, 3)
>>>np.indices((2,3))[0]
array([[0, 0, 0],
       [1, 1, 1]])
>>>np.indices((2,3))[1]
array([[0, 1, 2],
       [0, 1, 2]])
# 用于索引
>>>a=np.arange(6).reshape(2,3)
>>>b=np.indices((2,3))[0]
>>>a[b]
array([[[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]],
       [[3, 4, 5],
        [3, 4, 5],
        [3, 4, 5]]])
>>>c=np.indices((2,3))[1]
>>>a[:,c]
array([[[0, 1, 2],
        [0, 1, 2]],
       [[3, 4, 5],
        [3, 4, 5]]])
# 配合布尔索引进行mask操作
>>>a[b>0]
array([3, 4, 5])
>>>b>0
array([[False, False, False],
       [ True,  True,  True]])
```



**np.repeat:**对数组进行扩展

np.repeat(a,repeats,axis=None)

```python
>>>a=np.array([[10,20],[30,40]])
>>>np.repeat(a,[3,2],axis=0)
array([[10, 20],
       [10, 20],
       [10, 20],
       [30, 40],
       [30, 40]])
```



**np.tile:**对整个数组进行复制拼接

np.tile(a,reps)

```python
>>> a=np.array([10,20])
>>>np.tile(a, (3,2))
array([[10, 20, 10, 20],  
       [10, 20, 10, 20],  
       [10, 20, 10, 20]])  
```



**np.pad:**数组填充(padding)操作

np.pad(array,pad_width,mode,**kwags)

```python
>>>A = np.arange(95,99).reshape(2,2)
>>>np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0))
array([[ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, 95, 96,  0,  0,  0],
       [ 0,  0, 97, 98,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0]])
```



**np.flip:**沿着指定轴翻转

```python
>>>A = np.arange(4).reshape((2,2))
>>>np.flip(A,0)
array([[2, 3],
       [0, 1]])
```



**np.unravel_index:**返回indices的下标

np.unravel_index(indices,dims,order='C')

```python
>>>np.unravel_index([22, 41, 37], (7,6))
(array([3, 6, 6]), array([4, 5, 1]))
>>>a=np.array([[1,2,3],[4,3,2]])
>>>np.argmax(a)
3
>>>np.unravel_index(np.argmax(a),a.shape)
(1, 0)
```



**np.unique:**去除数组中重复数字，并进行排序后输出

```python
>>>a=np.array([[1,2,3],[4,3,2]])
>>>np.unique(a)
array([1, 2, 3, 4])
```



