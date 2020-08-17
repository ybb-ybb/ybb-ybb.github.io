---
date: 2020-08-17 21:49:00
description: python,列表、字符串和字典方法笔记
categories: 学习
title: python序列方法
author: 鱼摆摆
comments: true
tags: 
 - python
photos: https://w.wallhaven.cc/full/ym/wallhaven-ym1wp7.jpg
---



**注：有些方法在 python2 中不存在**



# 列表方法

通过列表推导式创建列表

```python
>>> x=[i**2 for i in range(3)]
>>> x
[0, 1, 4]
```



**append** : 将一个对象附加到列表末尾

**clear** ：就地清空列表内容

**copy** ： 复制列表

**count** ： 计算指定元素在列表中出现了多少次

**extend** ： 使用列表扩展另一个列表

**index** ： 查找指定值第一次出现的索引

**insert** ： 讲一个对象插入列表

**pop** ： 从列表删除一个元素(默认最后一个元素)，并返回这一元素

**remove** ： 删除第一个为指定值的元素

**reverse** ： 按相反的顺序排列列表中的元素

**sort** ： 对列表就地排序，接受两个可选参数key和values。

```python
>>> lst = [1, 2, 3]
>>> lst.append(4)
>>> lst
[1, 2, 3, 4]
>>> lst.clear()
>>> lst
[]
>>> a = [1, 2, 3]
>>> b=a.copy()
>>> b
[1, 2, 3]
>>> a.count(1)
1
>>> a.extend(b)
>>> a
[1, 2, 3, 1, 2, 3]
>>> a.index(2)
1
>>> a.insert(1,4)
>>> a
[1, 4, 2, 3, 1, 2, 3]
>>> a.pop()
3
>>> a
[1, 4, 2, 3, 1, 2]
>>> a.pop(0)
1
>>> a
[4, 2, 3, 1, 2]
>>> a.remove(1)
>>> a
[4, 2, 3, 2]
>>> a.reverse()
>>> a
[2, 3, 2, 4]
>>> a.sort()
>>> a
[2, 2, 3, 4]
>>> a.sort(key=lambda x : x % 3,reverse=True)
>>> a
[2, 2, 4, 3]

```



# 字符串方法

**center** ： 通过在字符串两边添加填充字符使字符串居中

**find** ：在字符串中查找子串，找到则返回子串第一个字符的索引，否则返回 -1(可指定起点值和终点值)

**join** ： 合并序列的元素，与 **split** 相反

**lower** ： 返回字符串的小写版本

**replace** ： 将指定子串都替换为另一个字符串，并返回替换后的结果

**split** ： 与 **join** 相反，将字符串拆分为序列

**strip** ： 将字符串开头和末尾的指定字符删除

**isspace** ： 是否是空格

**isdight** ： 是否是数字

**isupper** : 是否是大写字母

**islower** ： 是否是小写字母

```python
>>> x="Hello World!"
>>> x.center(15)
'  Hello World! '
>>> x.center(15,'*')
'**Hello World!*'
>>> x.find('llo')
2
>>> x.find('llo',3)
-1
>>> seq = ['1','2','3']
>>> '+'.join(seq)
'1+2+3'
>>> x.lower()
'hello world!'
>>> x.replace('hello','Hello')
'Hello World!'
>>> x.split()
['Hello', 'World!']
>>> x = x.center(15,'*')
>>> x
'**Hello World!*'
>>> x.strip('*')
'Hello World!'
>>> x
'**Hello World!*'
```





# 字典

创建字典：通过键——值对序列或者字典推导式创建

```python
>>> items = [('name','Gumby'),('age',42)]
>>> d=dict(items)
>>> d
{'name': 'Gumby', 'age': 42}
>>> d=dict(name='Gumby',age=42)
>>> d
{'name': 'Gumby', 'age': 42}
>>> x={i:i ** 2 for i in range(3)}
>>> x
{0: 0, 1: 1, 2: 4}
```



**clear ** ：删除所有字典项

**copy** ： 返回一个新字典，包含的键值对与原字典相同

**fromkeys** ：创建一个新字典，包含指定的键，且每个键对应的值都是None，可指定对应的默认值

**get** ：访问不存在的建时返回None，可指定返回值

**items** ： 返回一个包含所有字典项的列表，其中每个元素都是（key,value）的形式

**pop** ： 获取与指定键相关联的值，并将该键值对从字典中删除

**popitem** ： 随机弹出一个字典项

**setdefault** ： 类似**get** ，当字典中不包含指定的键时，在字典中添加指定的键值对

**update** ： 使用一个字典中的项来更新另一个字典

**keys** ： 返回一个字典视图，其中包含字典中的键

**values** ： 返回一个字典视图，其中包含字典中的值

```python
>>> d=dict(name='Gumby',age=42)
>>> d
{'name': 'Gumby', 'age': 42}
>>> d.clear()
>>> d
{}
>>> d=dict(name='Gumby',age=42)
>>> x=d.copy()
>>> x
{'name': 'Gumby', 'age': 42}
>>> {}.fromkeys(['name','age'])
{'name': None, 'age': None}
>>> {}.fromkeys(['name','age'],False)
{'name': False, 'age': False}
>>> x.get('score')
>>> x.items()
dict_items([('name', 'Gumby'), ('age', 42)])

IndentationError: expected an indented block
>>> for key,value in x.items():
...     print(value)
Gumby
42
>>> x
{'name': 'Gumby', 'age': 42}
>>> x.pop('name')
'Gumby'
>>> x.popitem()
('age', 42)
>>> x
{}
>>> d
{'name': 'Gumby', 'age': 42}
>>> d.setdefault('name','N/A')
'Gumby'
>>> d.setdefault('score','N/A')
'N/A'
>>> d
{'name': 'Gumby', 'age': 42, 'score': 'N/A'}
>>> x={'score':88}
>>> d.update(x)
>>> d
{'name': 'Gumby', 'age': 42, 'score': 88}
>>> x.keys()
dict_keys(['name', 'age'])
>>> x.values()
dict_values(['Gumby', 42])

```



```python

```







