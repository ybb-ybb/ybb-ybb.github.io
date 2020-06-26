---
date: 2020-05-28 18:49:00
description: 终端挂起和恢复-screen
categories: 瞎折腾
title: screen
author: 鱼摆摆
comments: true
tags: 
 - screen
photos: https://w.wallhaven.cc/full/13/wallhaven-13vym3.jpg
---

# 安装

```bash
$ sudo apt-get install screen
```

# 使用

```bash
# 创建一个名为screen-name的会话
$ screen -S screen-name
# 离开会话: ctrl+a+d
# 恢复创建的会话
$ screen -r screen-name
# 或，如果只有一个会话
$ screen -r
# 查看已经创建的会话
$ screen -ls
# 退出会话
$ exit
```

# 其它命令

> Ctrl + a，d 			#暂离当前会话
> Ctrl + a，c 			#在当前screen会话中创建一个子会话
> Ctrl + a，w			#子会话列表
> Ctrl + a，p			#上一个子会话
> Ctrl + a，n       	#下一个子会话
> Ctrl + a，0-9    	 #在第0窗口至第9子会话间切换

# 鼠标回滚

screen模式下，无法在终端中使用鼠标滚轴进行翻页，解决：

```tex
ctrl + a +[ 进入回滚模式
ctrl + c 切换回之前模式
```

