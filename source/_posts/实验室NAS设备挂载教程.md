---
date: 2020-11-21 20:22:00
description: 实验室NAS挂载教程
title: 实验室NAS挂载教程
author: 鱼摆摆
comments: true
tags: 
 - 服务器
photos: https://w.wallhaven.cc/full/m9/wallhaven-m9q71m.png
password: 123456
categories: 实验室
---

# 实验室NAS设备挂载教程

## Windows设备：

搜索Windows功能，打开NFS客户端

![image-20201121201631987](https://cdn.jsdelivr.net/gh/ybb-ybb/gallery/img/image-20201121201631987.png)

打开我的电脑——计算机——映射网络驱动器，填写如下：

![image-20201121201755473](https://cdn.jsdelivr.net/gh/ybb-ybb/gallery/img/image-20201121201755473.png)

此时会看到我的电脑中出现一个共享盘

![image-20201121201910285](https://cdn.jsdelivr.net/gh/ybb-ybb/gallery/img/image-20201121201910285.png)



## Mac和Ubuntu

在ubuntu上需要先安装nfs工具, mac不需要

```bash
$ sudo apt-get install nfs-common
```



在任意位置创建一个文件夹，用于挂载NAS

在终端中运行命令挂载

```bash
# 将NAS挂载到本地的/CG目录下
$ mount -t nfs 10.8.130.30:/CG /CG
```

![image-20201121202156739](https://cdn.jsdelivr.net/gh/ybb-ybb/gallery/img/image-20201121202156739.png)



