---
date: 2020-04-10 09:15:00
description: 一款内网穿透工具
categories: 瞎折腾
title: 内网穿透proxyer
author: 鱼摆摆
comments: true
tags: 
 - 内网穿透
photos: https://w.wallhaven.cc/full/r2/wallhaven-r2x92w.jpg
---

**Github地址：**<https://github.com/khvysofq/proxyer> 

# 服务端

安装docker：

```bash
$ curl -sSL https://get.docker.com/ | sh
$ systemctl start docker
$ systemctl enable docker
```

安装docker compose

```bash
$ curl -L "https://get.daocloud.io/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ chmod +x /usr/local/bin/docker-compose
```

安装proxyer：

```bash
$ wget https://raw.githubusercontent.com/khvysofq/proxyer/master/docker-compose.yml
# 后面1.1.1.1改成服务器ip地址
$ export PROXYER_PUBLIC_HOST=1.1.1.1
$ docker-compose up -d
```

安装完成后，通过`ip:6789`访问服务端`WEB`管理面板了，进去后需要设置一个客户端认证密码。 

![image.png](https://i.loli.net/2020/04/10/nlDFJKdtZx9aOQi.png)

# 客户端

从web管理面板下载对应系统客户端

windows系统直接运行，linux解压后运行`./proxyer`，按照提示浏览器进入`127.0.0.1:9876`

![image.png](https://i.loli.net/2020/04/10/g8iJZVDtvX6beqE.png)

如图，内网地址填`127.0.0.1:22`，序列号自定义

安装ssh，已安装可跳过：

```bash
$ sudo apt-get install ssh
```

以图片为例，远程ssh连接时，连接使用：

```bash
$ ssh -p 43537 yu@66.152.179.100
```

# 设置为开机自启

(支持Ubuntu16.04，18.04好像不支持这种方式了)

编辑`/etc/rc.local`文件，在`exit 0`前添加`proxyer`文件的位置

如：`~/proxyer`





