---
date: 2020-09-12 14:42:00
description: neptune可视化和实验监测
categories: 实验室
title: neptune
author: 鱼摆摆
comments: true
tags: 
 - 实验室
 - neptune
photos: https://w.wallhaven.cc/full/5w/wallhaven-5w26x1.jpg
---

# Getting started

[Neptune](https://neptune.ai)

- 比较和调试ML实验和模型
- 监测实验结果
- 保存模型、参数
- 监测GPU利用状态

## Install Neptune client library

```bash
$ pip install neptune-client
# or
$ conda install -c conda-forge neptune-client
```

## Copy and export your API token

在`~/.bashrc` 中添加

```bash
export NEPTUNE_API_TOKEN="your API token"
```

执行

```bash
$ source ~\.bashrc
```

## 测试

新建python文件，执行

```python
import neptune

# The init() function called this way assumes that
# NEPTUNE_API_TOKEN environment variable is defined.

neptune.init('ybb-ybb/sandbox')
neptune.create_experiment(name='minimal_example')

# log some metrics

for i in range(100):
    neptune.log_metric('loss', 0.95**i)

neptune.log_metric('AUC', 0.96)
```

```bash
$ bash main.py
```

![image-20200912103059671](https://cdn.jsdelivr.net/gh/ybb-ybb/gallery/img/image-20200912103059671.png)

# 使用

## create an experiment

**定义参数运行新实验**：

```python
# Define parameters

PARAMS = {'decay_factor' : 0.5,
          'n_iterations' : 117}

# Create experiment with defined parameters

neptune.create_experiment (name='example_with_parameters',
                          params=PARAMS)
```

**记录图像**：

```python
# Log image data

import numpy as np

array = np.random.rand(10, 10, 3)*255
array = np.repeat(array, 30, 0)
array = np.repeat(array, 30, 1)
neptune.log_image('mosaics', array)
```

**记录文本**

```python
# Log image data

import numpy as np

array = np.random.rand(10, 10, 3)*255
array = np.repeat(array, 30, 0)
array = np.repeat(array, 30, 1)
neptune.log_image('mosaics', array)
```

**保存模型**

```python
# log some file

# replace this file with your own file from local machine
neptune.log_artifact('model_weights.pkl')

# log file to some specific directory (see second parameter below)

# replace this file with your own file from local machine
neptune.log_artifact('model_checkpoints/checkpoint_3.pkl', 'training/model_checkpoints/checkpoint_3.pkl')
```

**上传代码**

```python
# Upload source code

# replace these two source files with your own files.
neptune.create_experiment(upload_source_files=['main.py', 'model.py'])
```

**标签**

```python
# add tag when experiment is created
neptune.create_experiment(tags=['training'])

# add single tag
neptune.append_tag('transformer')

# add few tags at once
neptune.append_tags('BERT', 'ELMO', 'ideas-exploration')
```

