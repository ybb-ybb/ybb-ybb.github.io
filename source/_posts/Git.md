---
date: 2020-03-30 18:49:00
description: Git的一点教程
categories: 瞎折腾
title: Git！
author: 鱼摆摆
comments: true
tags: 
 - Git
photos: https://th.wallhaven.cc/lg/ox/oxv6gl.jpg
---

# 工作原理/流程

![image.png](https://i.loli.net/2020/03/30/NPolTDqks2tdhKR.png)

- Workspace：工作区
- Index / Stage：暂存区
- Repository：仓库区（或本地仓库）
- Remote：远程仓库

# 配置

在Github注册后，做一下本地的Git仓库配置：

```bash
$ git config --global user.name "ybb-ybb"
$ git config --global user.email "21901037@mail.dlut.edu.cn"
```

如果对某个仓库使用不同的用户名和邮箱，去掉`--global`参数即可

# 本地仓库

## 提交

```bash
# 将当前目录变成git可管理的仓库
$ git init
# 将readme.md添加到缓存区
$ git add readme.md
# 将所有文件添加到缓存区
$ git add .
# 将缓存区文件提交到仓库
$ git commit -m "提交信息"
# 查看是否还有文件未提交
$ git status
# 查看文件修改内容
$ git diff readme.md
# 查看历史记录
$ git log
# 查看历史记录的简单信息
$ git log –pretty=oneline
```

## 版本回退

```bash
# 回到上一个版本
$ git reset --hard HEAD^
# 同理，回到上上个版本等，以此类推
$ git reset --hard HEAD^^
# 或
$ git reset --hard HEAD~2
# 查看版本号变化
$ git reflog
# 版本号回退
$ git reset --hard {版本号}
# 丢弃工作区的修改（回到暂存区的状态）
$ git checkout -- filename
```

# 远程仓库

## 配置

本地Git仓库和Github仓库之间是通过ssh加密的，因此需要先进行一些配置：

第一步：创建SSH Key。在主目录下，看看有没有.ssh目录，如果有，再看看这个目录下有没有id_rsa和id_rsa.pub这两个文件，如果有的话，直接跳过此如下命令，如果没有的话：

```bash
ssh-keygen -t rsa –C “youremail@example.com”
```

id_rsa是私钥，id_rsa.pub是公钥

第二步：登录github,打开” settings”中的SSH Keys页面，然后点击“Add SSH Key”,填上任意title，在Key文本框里黏贴id_rsa.pub文件的内容 ，Add key

![](https://mmbiz.qpic.cn/mmbiz_png/e1jmIzRpwWiaEynpFwWSmr59icj386rKKx9HfRIwgwuTkiaggs8OS1CZYHGMpnKVx6Yl2bicM8s9NGb69hrVMziaBAQ/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 推送到远程仓库

```bash
# 先创建一个github仓库，复制http或ssh地址,然后关联
$ git remote add origin {http adress}
# 推送，第一次推送时添加-u参数,把master分支推送到远程
$ git push -u origin master
```

## 分支

```bash
# 创建分支
$ git branch backup
# 切换分支
$ git checkout backup
# 或者：创建+切换分支
$ git checkout -b backup
# 查看当前分支
$ git branch
# 在master分支上，将分支backup合并到master上
$ git merge backup
# 删除分支
$ git branch -d backup
```

