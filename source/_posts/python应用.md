---
date: 2020-12-06 11:38:00
description: 常见的python应用
title: python应用
author: 鱼摆摆
comments: true
tags: 
 - python
photos: https://w.wallhaven.cc/full/nm/wallhaven-nmwp71.jpg
categories: 学习
---
# collections

## defaultdict

defaultdict解决的是dict中常见的key为空的情况

```python
from collections import defaultdict
d = defaultdict(list)

for k, v in data:
    d[k].append(v)
```

使用defaultdict之后，如果key不存在，会自动返回预先设置的默认值。defaultdict传入的默认值可以使一个类型或一个方法。

## Counter

计数和排序功能

```python
words = ['apple', 'apple', 'pear', 'watermelon', 'pear', 'peach']
from collections import Counter
counter = Counter(words)

>>> print(counter)
Counter({'apple': 2, 'pear': 2, 'watermelon': 1, 'peach': 1})
```

Counter也提供了most_common方法，可以筛选topk

```python
>>> counter.most_common(1)
[('apple', 2)]
```

## deque

双端队列，支持队首和队尾的元素插入和弹出

常用API：`clear,copy,count,extend,append,pop,appendleft,popleft`

```python
from collections import deque
dque = deque(maxlen=10)
# 假设我们想要从文件当中获取最后10条数据
for i in f.read():
    dque.append(i)
```

## Orderdict

python2中的字典类型为无序，但python3中为有序。

Orderdict是有序字典



# Heapq

全称: heqp queue，即堆和队列

## nlargest和nsmallest：筛选topk

```python
import heapq

nums = [14, 20, 5, 28, 1, 21, 16, 22, 17, 28]
heapq.nlargest(3, nums)
# [28, 28, 22]
heapq.nsmallest(3, nums)
# [1, 5, 14]
```

可以通过关键词参数key自定义排序规则

```python
laptops = [
    {'name': 'ThinkPad', 'amount': 100, 'price': 91.1},
    {'name': 'Mac', 'amount': 50, 'price': 543.22},
    {'name': 'Surface', 'amount': 200, 'price': 21.09},
    {'name': 'Alienware', 'amount': 35, 'price': 31.75},
    {'name': 'Lenovo', 'amount': 45, 'price': 16.35},
    {'name': 'Huawei', 'amount': 75, 'price': 115.65}
]

cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
```

##  优先队列

heapq.heapify方法输入一个数组，返回的结果是这个数组生成的堆(等价于优先队列)

```python
heap = [5,8,0,3,6,7,9,1,4,2]
hq.heapify(heap)
>>> heap
[0, 1, 5, 3, 2, 7, 9, 8, 4, 6]
```

可以直接使用heapq的heappop方法和heappush方法维护这个堆。

heapq.push传入两个参数，一个是存储元素的数组，另一个是要存储的元素。

时间复杂度：nlogn



# 自定义排序

## 字典排序

```python
kids = [
    {'name': 'xiaoming', 'score': 99, 'age': 12},
    {'name': 'xiaohong', 'score': 75, 'age': 13},
    {'name': 'xiaowang', 'score': 88, 'age': 15}
]
# 按照score来排序
sorted(kids, key=lambda x: x['score'])
# 多关键词排序
sorted(kids, key=lambda x: (x['score'], x['age']))
```

python中自带的库可以代替匿名函数

```python
from operator import itemgetter

sorted(kids, key=itemgetter('score'))
sorted(kids, key=itemgetter('score', 'age'))
```

## 对象排序

```python
class Kid:
    def __init__(self, name, score, age):
        self.name = name
        self.score = score
        self.age = age

    def __repr__(self):
        return 'Kid, name: {}, score: {}, age:{}'.format(self.name, self.score, self.age)
# 为了方便观察打印结果，我们重载了__repr__方法，可以指定在print的时候的输出结果
```

```python
from operator import attrgetter
kids = [Kid('xiaoming', 99, 12), Kid('xiaohong', 75, 13), Kid('xiaowang', 88, 15)]

sorted(kids, key=attrgetter('score'))
sorted(kids, key=lambda x: x.score)
```

## 自定义排序

```python
# 若x<y,返回一个负数，若x>y,返回一个正数，否则返回0
def cmp(kid1, kid2):
    if kid1.score == kid2.score:
        return kid1.age - kid2.age
    else:
        return kid1.score - kid2.score
      
from functools import cmp_to_key

sorted(kids, key=cmp_to_key(cmp))
```

也可以在类中重载比较函数`__lt__`，更改其默认排序方式

```python
lass Kid:
    def __init__(self, name, score, age):
        self.name = name
        self.score = score
        self.age = age

    def __repr__(self):
        return 'Kid, name: {}, score: {}, age:{}'.format(self.name, self.score, self.age)

    def __lt__(self, other):
        return self.score > other.score or (self.score == other.score and self.age < other.age)
```



# map,reduce和filter

## map

map执行的是一个映射，会将一个序列通过一个函数映射到另一个序列，从而避免显式迭代。

```python
def dis(point):
    return math.sqrt(point[0]**2 + point[1]**2)
points = [[0, 1], [2, 4], [3, 2]]
# map的输出值是一个迭代器，将其转换为list类型
print(list(map(dis, points)))
# 用匿名函数简化
map(lambda x: math.sqrt(x[0]**2 + x[1] ** 2), points)
```

## reduce

reduce是依次调用，将两个元素归并成一个结果

reduce(f, [a, b, c, d])的返回值为f(f(f(a, b), c), d)

```python
from functools import reduce

def f(a, b):
    return a + b
    
print(reduce(f, [1, 2, 3, 4]))
10
```

## filter

过滤掉所有为False的元素

```python
arr = [1, 3, 2, 4, 5, 8]
list(filter(lambda x: x % 2 > 0, arr))
```

## compress

根据一个序列的条件过滤另一个序列

```python
student = ['xiaoming', 'xiaohong', 'xiaoli', 'emily']
scores = [60, 70, 80, 40]
from itemtools import compress

>>> pass = [i > 60 for i in scores]
>>> print(pass)
[False, True, True, False]

>>> list(compress(student, pass))
['xiaohong', 'xiaoli']
```



# 生成器和迭代器

### 容器迭代器

对于一个可迭代对象(python中的tuple,list,dict等都是可迭代对象)，可以用关键字`iter`获得一个相应的迭代器。

```python
arr = [1, 3, 4, 5, 9]
it = iter(arr)
print(next(it))
print(next(it))
```

当越界时会抛出`StopIteration` 的Error。

也可以使用for去进行迭代。

### 自己创建迭代器

在定义类时，添加`__iter__`和`__next__`方法，其中`__iter__`方法用来初始化并返回迭代器。(iterable 的 `__iter__` 返回 iterator, iterator本身也是一个iterable对象)

```python
class PowTwo:
    """Class to implement an iterator
    of powers of two"""

    def __init__(self, max = 0):
        self.max = max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.max:
            result = 2 ** self.n
            self.n += 1
            return result
        else:
            raise StopIteration
```

```python
>>> a = PowTwo(4)
>>> i = iter(a)
>>> next(i)
1
>>> next(i)
2
>>> next(i)
4
>>> next(i)
8
>>> next(i)
16
>>> next(i)
Traceback (most recent call last):
...
StopIteration

>>> for i in PowTwo(5):
...     print(i)
... 
1
2
4
8
16
32
```

## 生成器

### 括号创建法

```python
g = (i * 3for i in range(10))
print(next(g))
```

### 函数创建法

```python
def test():
    n = 0
    whileTrue:
        if n < 3:
            yield n
            n += 1
        else:
            yield10
            
            
if __name__ == '__main__':
    t = test()
    for i in range(10):
        print(next(t))
```

二叉树的遍历：

```python
class Node:

    def __init__(self, key):
        self.key = key
        self.lchild = None
        self.rchild = None
        self.iterated = False
        self.father = None

    def iterate(self):
        if self.lchild isnotNone:
            yieldfrom self.lchild.iterate()
        yield self.key
        if self.rchild isnotNone:
            yieldfrom self.rchild.iterate()
```



# Itertools

## 跳过迭代器开头

```python
# 跳过头部注释#的部分
from itertools import dropwhile
with open('xxxx.txt') as f:
    for line in dropwhile(lambda line: line.startswith('#'), f):
        print(line)
# 从第三行开始
from itertools import dropwhile
with open('xxxx.txt') as f:
    for line in islice(f, 3, None):
        print(line)
```

## 排列组合

```python
# 排列
items = ['a', 'b', 'c']
from itertools import permutations

for p in permutations(items):
    print(p)
# 只保留前两个元素
for p in permutations(items, 2):
    print(p)
# 组合
from itertools import combindations
for c in combinations(items):
    print(c)
# 有放回的组合
for c in combinations_with_replacement(items, 3):
    print(c)
```

## 迭代的合并

```python
from itertools import chain
nums = [1, 2, 3]
chars = ['a', 'b', 'c']

for i in chain(nums, chars):
    print(i)
```



# 打印对象

直接打印实例时，会返回一个内存地址：

```python
class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


if __name__ == "__main__":
    p = point(3, 4)
    print(p)
<__main__.point object at 0x10a18c210>
```

重置`__str__`方法：

```python
class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return 'x: %s, y: %s' % (self.x, self.y)
x: 3, y: 4
```

重置`__repr__`方法

```python
class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'x: %s, y: %s' % (self.x, self.y)
x: 3, y: 4
```

注：`__str__` 侧重于展示，`__repr__` 侧重于交互式中的报告

## format方法

```python
print('x:{x},y:{y}'.format(x=3,y=4))
```

重载`__format__`函数：

```python
class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return 'x: %s, y: %s' % (self.x, self.y)

    def __format__(self, code):
        return 'x: {x}, y: {y}'.format(x = self.x, y = self.y)
print('The point is {}'.format(p))
The point is x: 3, y: 4
```



# BiseCt模块进行二分查找

| 名称           | 功能                                         |
| -------------- | -------------------------------------------- |
| bisect_left()  | **查找**目标元素左侧插入点                   |
| bisect_right() | **查找**目标元素右侧插入点                   |
| bisect()       | 同 bisect_right()                            |
| insort_left()  | 查找目标元素左侧插入点，并保序地**插入**元素 |
| insort_right() | 查找目标元素右侧插入点，并保序地**插入**元素 |
| insort()       | 同 insort_right()                            |



# Any, All

```python
>>> any([0,1])
True
>>> any([0,'0',''])
True
>>> all(['a', 'b', 'c', 'd'])
True
>>> all(['a', 'b', '', 'd'])
False
```





# 正则表达式

## 字符匹配

在正则表达式中，`\d`可以匹配一个数字，`\s`可以匹配一个空格，`\w`可以匹配一个字母或数字，`.`可以匹配任意字符，`*`可以匹配任意个字符(包括0个)，`+`表示至少一个字符，`{n}`表示n个字符，`{n,m}`表示n-m个字符。

例如：`\d{3}\s+\d{3,8}`匹配  [三个数字+至少一个空格+3—8个数字]

## 精确匹配

要做更精确地匹配，可以用`[]`表示范围，比如：

- `[0-9a-zA-Z\_]`可以匹配一个数字、字母或者下划线；
- `[0-9a-zA-Z\_]+`可以匹配至少由一个数字、字母或者下划线组成的字符串，比如`'a100'`，`'0_Z'`，`'Py3000'`等等；
- `[a-zA-Z\_][0-9a-zA-Z\_]*`可以匹配由字母或下划线开头，后接任意个由一个数字、字母或者下划线组成的字符串，也就是Python合法的变量；
- `[a-zA-Z\_][0-9a-zA-Z\_]{0, 19}`更精确地限制了变量的长度是1-20个字符（前面1个字符+后面最多19个字符）。

`A|B`可以匹配A或B，所以`(P|p)ython`可以匹配`'Python'`或者`'python'`。

`^`表示行的开头，`^\d`表示必须以数字开头。

`$`表示行的结束，`\d$`表示必须以数字结束。

## re模块

先看看如何判断正则表达式是否匹配：

```
>>> import re
>>> re.match(r'^\d{3}\-\d{3,8}$', '010-12345')
<_sre.SRE_Match object; span=(0, 9), match='010-12345'>
>>> re.match(r'^\d{3}\-\d{3,8}$', '010 12345')
>>>
```

`match()`方法判断是否匹配，如果匹配成功，返回一个`Match`对象，否则返回`None`。常见的判断方法就是：

```
test = '用户输入的字符串'
if re.match(r'正则表达式', test):
    print('ok')
else:
    print('failed')
```



**切分字符串：**

用正则表达式切分字符串比用固定的字符更灵活

```
>>> 'a b   c'.split(' ')
['a', 'b', '', '', 'c']
```

```
>>> re.split(r'\s+', 'a b   c')
['a', 'b', 'c']
```

```
>>> re.split(r'[\s\,]+', 'a,b, c  d')
['a', 'b', 'c', 'd']
```

```
>>> re.split(r'[\s\,\;]+', 'a,b;; c  d')
['a', 'b', 'c', 'd']
```



**分组**

除了简单地判断是否匹配之外，正则表达式还有提取子串的强大功能。用`()`表示的就是要提取的分组（Group）。比如：

`^(\d{3})-(\d{3,8})$`分别定义了两个组，可以直接从匹配的字符串中提取出区号和本地号码：

```
>>> m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
>>> m
<_sre.SRE_Match object; span=(0, 9), match='010-12345'>
>>> m.group(0)
'010-12345'
>>> m.group(1)
'010'
>>> m.group(2)
'12345'
```

如果正则表达式中定义了组，就可以在`Match`对象上用`group()`方法提取出子串来。

注意到`group(0)`永远是原始字符串，`group(1)`、`group(2)`……表示第1、2、……个子串。

```
>>> t = '19:05:30'
>>> m = re.match(r'^(0[0-9]|1[0-9]|2[0-3]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])$', t)
>>> m.groups()
('19', '05', '30')
```

这个正则表达式可以直接识别合法的时间。但是有些时候，用正则表达式也无法做到完全验证，比如识别日期：

```
'^(0[1-9]|1[0-2]|[0-9])-(0[1-9]|1[0-9]|2[0-9]|3[0-1]|[0-9])$'
```

对于`'2-30'`，`'4-31'`这样的非法日期，用正则还是识别不了，或者说写出来非常困难，这时就需要程序配合识别了。



**贪婪匹配**

正则匹配默认是贪婪匹配，也就是匹配尽可能多的字符。举例如下，匹配出数字后面的`0`：

```
>>> re.match(r'^(\d+)(0*)$', '102300').groups()
('102300', '')
```

由于`\d+`采用贪婪匹配，直接把后面的`0`全部匹配了，结果`0*`只能匹配空字符串了。

必须让`\d+`采用非贪婪匹配（也就是尽可能少匹配），才能把后面的`0`匹配出来，加个`?`就可以让`\d+`采用非贪婪匹配：

```
>>> re.match(r'^(\d+?)(0*)$', '102300').groups()
('1023', '00')
```

**编译**

```
>>> import re
# 编译:
>>> re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')
# 使用：
>>> re_telephone.match('010-12345').groups()
('010', '12345')
>>> re_telephone.match('010-8086').groups()
('010', '8086')
```



# 装饰器

装饰器实质上是一个高阶函数

```python
from functools import wraps
def wrapexp(func):
    def wrapper(*args, **kwargs):
        print('this is a wrapper')
        func(*args, **kwargs)
    return wrapper

@wrapexp
def exp(a, b, c='3', d='f'):
    print(a, b, c, d)

>>>args = [1, 3]
>>>dt = {'c': 4, 'd': 5}
>>>exp(*args, **dt)

this is a wrapper
1 3 4 5
```

在这个例子当中，我们定义了一个wrapexp的装饰器。我们在其中的**wrapper方法当中实现了装饰器的逻辑**，wrapexp当中传入的参数func是一个函数，wrapper当中的参数则是func的参数。所以我们在wrapper当中调用func(*args, **kw)，就是调用打上了这个注解的函数本身。



利用装饰器计算函数耗时：

```python
import time
from functools import wraps
def timethis(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper
```



调用装饰器后，函数的元信息会丢失：

```python
from functools import wraps
def wrapexp(func):
    def wrapper(*args, **kwargs):
        print('this is a wrapper')
        func(*args, **kwargs)
    return wrapper

@wrapexp
def exp(a, b, c='3', d='f'):
    print(a, b, c, d)
    
>>> exp.__name__
'wrapexp'
```

Python当中为我们提供了一个专门的装饰器器用来保留函数的元信息，我们只需要在实现装饰器的wrapper函数当中**加上一个注解wraps**即可

```python
def wrapexp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('this is a wrapper')
        func(*args, **kwargs)
    return wrapper
```





# divmod

把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)。



# chr和ord

`ord()` 函数主要用来返回对应字符的ascii码，`chr()` 主要用来表示ascii码对应的字符他的输入时数字。



# 位运算



# 缓存机制与lru_cache

