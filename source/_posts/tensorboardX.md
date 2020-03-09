---
data: 2020/2/24 20:42:11
abstract: 用TensorBoardX进行pytorch训练的可视化
description: 用TensorBoardX进行pytorch训练的可视化
title: tensorboardX
author: 鱼摆摆
comments: true
tags: 
 - neural ode
photos: https://cdn.jsdelivr.net/gh/ybb-ybb/CDNrepository@1.1/img/fig.jpg
categories: 学习
---

# 创建

TensorBoardX的GitHub地址：[传送门](https://github.com/lanpa/tensorboardX)

首先创建一个 SummaryWriter 的示例 ：

```python
from tensorboardX import SummaryWriter

# Creates writer1 object.
# The log will be saved in 'runs/exp'
writer1 = SummaryWriter('runs/exp')

# Creates writer2 object with auto generated file name
# The log directory will be something like 'runs/Aug20-17-20-33'
writer2 = SummaryWriter()

# Creates writer3 object with auto generated file name, the comment will be appended to the filename.
# The log directory will be something like 'runs/Aug20-17-20-33-resnet'
writer3 = SummaryWriter(comment='resnet')
```

以上展示了三种初始化 SummaryWriter 的方法：

1. 提供一个路径，将使用该路径来保存日志
2. 无参数，默认将使用 `runs/日期时间` 路径来保存日志
3. 提供一个 comment 参数，将使用 `runs/日期时间-comment` 路径来保存日志

在浏览器中查看这些可视化数据：

```bash
tensorboard --logdir=<your_log_dir>
```

# 用各种add方法记录数据

## 数字(scalar)

```python
add_scalar(tag, scalar_value, global_step=None, walltime=None)
```

参数:

- tag (string): 数据名称，不同名称的数据使用不同曲线展示
- scalar_value (float): 数字常量值
- global_step (int, optional): 训练的 step
- walltime (float, optional): 记录发生的时间，默认为 time.time()

需要注意，这里的 scalar_value 一定是 float 类型，如果是 PyTorch scalar tensor，则需要调用 .item() 方法获取其数值。我们一般会使用 add_scalar 方法来记录训练过程的 loss、accuracy、learning rate 等数值的变化，直观地监控训练过程。

示例：

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i**2, global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)
```

## 图片(image)

需要pillow库的支持

用`add_image`记录单个图像数据，用`add_images`记录多个图像数据

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

`CHW`为channel\*hight\*width

示例：

```python
from tensorboardX import SummaryWriter
import cv2 as cv

writer = SummaryWriter('runs/image_example')
for i in range(1, 6):
    writer.add_image('countdown',
                     cv.cvtColor(cv.imread('{}.jpg'.format(i)), cv.COLOR_BGR2RGB),
                     global_step=i,
                     dataformats='HWC')
```

## 直方图(histogram)

```python
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None) 
```

示例：

```pythpn
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter('runs/embedding_example')
writer.add_histogram('normal_centered', np.random.normal(0, 1, 1000), global_step=1)
writer.add_histogram('normal_centered', np.random.normal(0, 2, 1000), global_step=50)
writer.add_histogram('normal_centered', np.random.normal(0, 3, 1000), global_step=100)
```

"DISTRIBUTIONS"和"HISTOGRAMS"两栏都是用来观察数据分布的。其中在"HISTOGRAMS"中，同一数据不同 step 时候的直方图可以上下错位排布 (OFFSET) 也可重叠排布 (OVERLAY)。 

## 运行图(graph)

```pythpn
add_graph(model, input_to_model=None, verbose=False, **kwargs)
```

可以可视化神经网络的结构，参考Github[官方样例](https://github.com/lanpa/tensorboardX/blob/master/examples/demo_graph.py )

## 嵌入张量(embedding)

使用 `add_embedding` 方法可以在二维或三维空间可视化 embedding 向量。 

```python
add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
```

参数：

- mat (torch.Tensor or numpy.array): 一个MxN矩阵，每行代表特征空间的一个数据点
- metadata (list or torch.Tensor or numpy.array, optional): 一个一维列表N，mat 中每行数据的 label，大小应和 mat 行数相同
- label_img (torch.Tensor, optional): 一个形如 NxCxHxW 的张量，对应 mat 每一行数据显示出的图像，N 应和 mat 行数相同
- global_step (int, optional): 训练的 step
- tag (string, optional): 数据名称，不同名称的数据将分别展示

示例：

```python
from tensorboardX import SummaryWriter
import torchvision

writer = SummaryWriter('runs/embedding_example')
mnist = torchvision.datasets.MNIST('mnist', download=True)
writer.add_embedding(
    mnist.train_data.reshape((-1, 28 * 28))[:100,:], #直接将mnist前100个数据展开成一维向量作为embedding
    metadata=mnist.train_labels[:100], #每个embedding的label
    label_img = mnist.train_data[:100,:,:].reshape((-1, 1, 28, 28)).float() / 255, #每个图像
    global_step=0
)
```

