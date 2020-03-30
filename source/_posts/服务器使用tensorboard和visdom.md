---
date: 2020-01-28 17:13::00
abstract: 服务器使用tensorboard和visdom
description: 服务器服务器使用tensorboard和visdom教程
title: 服务器使用tensorboard和visdom
author: 鱼摆摆
comments: true
tags: 
 - 服务器
 - 实验室
photos: https://cn.bing.com/th?id=OIP.bDfxJo0jUHFW9aIYlvoG_AHaFb&pid=Api&rs=1
categories: 实验室
---
# 服务器使用tensorboard和visdom

## 以tensorboard为例：

创建容器时开放6006端口

```bash
# 运行容器时将服务器docker容器的6006端口暴漏到自己主机ip下的16006端口(可自己指定)
$ docker run -p <ip>:16006:6006 -it -v /data:/workspace/data --runtime=nvidia --net=host --name=temp /bin/bash
```
或者
```bash
# 在连接ssh时，将docker容器中的6006端口重新定向到自己机器上
$ ssh -p 1001 -L 16006:<ip>:6006 root@10.7.60.40
```
