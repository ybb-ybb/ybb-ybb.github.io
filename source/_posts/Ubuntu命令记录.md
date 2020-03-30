---
date: 2020-01-11 20:32:00
description: NVIDIA,CUDA,CUDNN,Anaconda
title: NVIDIA,CUDA,CUDNN,Anaconda
author: 鱼摆摆
comments: true
tags: 
 - nvidia
 - cuda
 - cudnn
 - anaconda
photos: https://gss0.baidu.com/-4o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/7af40ad162d9f2d3b3f12a16aeec8a136227cc63.jpg
categories: 瞎折腾
---
# NVIDIA,CUDA,CUDNN
## ppa安装NVIDIA驱动
```bash
  $ sudo add-apt-repository ppa:graphics-drivers/ppa
  $ sudo apt-get update
  $ ubuntu-drivers devices
  $ sudo apt-get install nvidia-driver-xxx
```
## 自动安装NVIDIA驱动

```bash
# 卸载残余驱动
sudo apt-get --purge remove "*nvidia*"
# 查看推荐驱动版本
ubuntu-drivers devices
# 自动安装
sudo ubuntu-drivers autoinstall
```



 ## .deb安装CUDA

 [ 下载deb文件](https://developer.nvidia.com/cuda-downloads)
```'bash
$ sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
$ sudo apt-get update
$ sudo apt-get install cuda
```
 ## 安装CUDNN
 [下载符合自己cuda版本的cudnn](https://developer.nvidia.com/rdp/cudnn-archive)
 ### 安装cudnn
 安装过程实际是将cudnn的头文件复制到CUDA的头文件目录里
 ```bash
 $ sudo cp cuda/include/* /usr/local/cuda-10.0/include/
 $ sudo cp cuda/lib64/* /usr/local/cuda-10.0/lib64/
 # 添加可执行权限
 $ sudo chmod +x /usr/local/cuda-10.0/include/cudnn.h
 $ sudo chmod +x /usr/local/cuda-10.0/lib64/libcudnn*
 ```
 ### 检验
 ```bash
 $ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
 ```
 ## 指定运行程序使用的GPU
 在程序中添加
 ```python
 import os
 os.environ['CUDA_VISIBLE_DEVICES]='0'
 ```
 或者在终端中
 ```bash
 $ CUDA_VISIBLE_DEVICES=0 python main.py
 ```
 ## 命令行安装Anaconda
 ```bash
 $ wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
 $ bash Anaconda3-5.0.1-Linux-x86_64.sh
 # 添加环境变量，可选
 $ echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc
 $ source .bashrc
 ```
