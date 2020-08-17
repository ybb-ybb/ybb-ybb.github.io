---
date: 2020-06-26 18:32:00
description: 利用proxychains在终端使用socks5代理
categories: 瞎折腾
title: 终端使用socks5代理
author: 鱼摆摆
comments: true
tags: 
 - 科学上网
 - 代理
photos: https://w.wallhaven.cc/full/6k/wallhaven-6k3oox.jpg
---

# proxychains安装

```bash
# git仓库中编译安装
$ git clone https://github.com/rofl0r/proxychains-ng.git
$ cd proxychains-ng
$ ./configure
$ make && make install
$ cp ./src/proxychains.conf /etc/proxychains.conf
$ cd .. && rm -rf proxychains-ng
# 或直接安装
$ brew install proxychains-ng
```

# 编辑proxychains配置

```bash
$ vim /etc/proxychains.conf
```

将`sock4 127.0.0.1 9095` 改为`socks5 12.0.0.1 1080` (配置同ssr或v2ray客户端一致)

# 使用方法

在需要代理的命令前加上 `proxychains4` ，如

```bash
$ proxychains4 wget http://xxx.com/xxx.zip
```

 