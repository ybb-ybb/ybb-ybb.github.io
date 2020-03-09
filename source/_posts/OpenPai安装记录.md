---
date: 2020-01-16 10:48:00
description: OpenPAI安装记录
title: OpenPAI安装记录
author: 鱼摆摆
comments: true
tags: 
 - openpai
photos: https://cdn.jsdelivr.net/gh/ybb-ybb/CDNrepository@1.1/img/fig.jpg
categories: 瞎折腾
---
# OpenPAI安装记录
## 环境准备
- 一台master主机和多台worker主机，一台维护机
- 所有节点不要安装CUDA驱动，具有统一的登录账户和密码
- 开启ssh功能和ntp功能(互相访问，时间同步)
## 部署过程
### 安装docker-ce
```bash
$ sudo apt-get -y install docker.io 
$ sudo docker pull docker.io/openpai/dev-box:v0.14.0
```
### 运行dev-box
```bash
$ sudo docker run -itd \
        -e COLUMNS=$COLUMNS -e LINES=$LINES -e TERM=$TERM \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /pathConfiguration:/cluster-configuration  \
        -v /hadoop-binary:/hadoop-binary  \
        --pid=host \
        --privileged=true \
        --net=host \
        --name=dev-box \
        docker.io/openpai/dev-box:v0.14.0
```
  ### 登录dev-box
  ```bash
 $ sudo docker exec -it dev-box /bin/bash
 $ cd /pai/deployment/quick-start/
  ```
 ### 修改配置信息
 ```bash
 $ cp quick-start-example.yaml quick-start.yaml
 $ vim quick-start.yaml
 ```
 修改内容：
 ```vim
 	machines:
  - <ip-of-master>
  - <ip-of-worker1>
  - <ip-of-worder2>

 ssh-username: <username>
 ssh-password: <password>
 ```
 ### 生成OepnPai配置文件
 ```bash
 $ cd /pai
 $ python paictl.py config generate -i /pai/deployment/quick-start/quick-start.yaml -o ~/pai-config -f 
 $ cd ~/pai-config/
 ```
 ### 修改kubernetes-configuration.yaml  
 将docker-registry替换为国内镜像库
 ```vim
 docker-registry: docker.io/mirrorgooglecontainers
 ```
 ### 修改layout.yaml  
 修改自己机器的配置信息
 ```vim
   machine-sku:
      GENERIC:
         mem: 256G
         gpu:
             type: TITAN V
             count: 1
         cpu:
             vcore: 4
         os: ubuntu16.04
       Worker1:
         mem: 256G
         gpu:
             type: GeForce RTX 2080Ti
             count: 4
         cpu:
             vcore: 4
         os: ubuntu16.04
       Worker2:
         mem: 256G
         gpu:
             type: GeForce RTX 2080Ti
             count: 4
         cpu:
             vcore: 4
         os: ubuntu16.04
 ```
 ### 修改services-configuration.yaml  
 解除common和data-path两个字段的注释，将data-path赋值到真实位置，作为服务数据存储路径
 ```bash
 cluster:
  common:
  #  cluster-id: pai-example
  #
  #  # HDFS, zookeeper data path on your cluster machine.
    data-path: "/data"
 ```
 tag字段修改为真实版本 v014.0  
 可修改cluster-id,后面会用到  
 修改rest-server下的用户名和密码，作为登录平台的账户密码

 指定显卡驱动版本，不指定的话默认安装384.11，这个驱动是不支持图灵核心显卡的，安装到后面会出现'nvidia-drm' not found 错误，驱动版本只能从注释里的版本选择
 ```vim
 drivers:
  set-nvidia-runtime: false
  # You can set drivers version here. If this value is miss, default value will be 384.111
  # Current supported version list
  # 384.111
  # 390.25
  # 410.73
  version: "410.73"
 ```
  ### 部署Kubernetes
  http:<master-ip>:9090查看进度
  ```bash
  $ cd /pai
  $ python paictl.py cluster k8s-bootup -p ~/pai-config
  ```
  ### 更改配置文件到kubernetes
  ```bash
  $ cd /pai
  python paictl.py config push -p ~/pai-config/ -c ~/.kube/config
  ```
  若报错，卸载openpai组件和ks组件，检查之前的配置文件，重新安装
  ```bash
  $ python paictl.py service [delete|start|stop] -c ~/.kube/config [-n name]  
  # 卸载openpai组件
  $ python paictl.py service delete -c ~/.kube/config 
  # 卸载k8s组件
  $ python paictl.py cluster k8s-clean -p ~/pai-config/ 
  ```
  ### 启动Openpai
  ```bash
  $ python paictl.py service start -c ~/.kube/config
  ```
  ### 界面
```http
http://<master-ip>:9090  
http://<master-ip>:80
```
## Reference
<https://github.com/kangapp/openPAI>  
<https://github.com/microsoft/pai>  
<https://zhuanlan.zhihu.com/p/64061072>
