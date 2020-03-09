---
date: 2020-02-10 11:16:00
description: emacs
title: emacs基本操作
author: 鱼摆摆
comments: true
tags: 
 - emacs
photos: https://cdn.jsdelivr.net/gh/ybb-ybb/CDNrepository@1.1/img/fig.jpg
categories: 瞎折腾
---
# Emacs基本操作

C = Ctrl, M = Alt

##  光标移动

| C-v 向下翻页   | M-v 向上翻页      |
| -------------- | ----------------- |
| C-b 向左(back) | C-f 向右(forward) |
| C-n 向下(next)           |      C-p 向上(previous)   |
|  M-b 上一个单词        |      M-f 下一个单词   |
|  C-a 行首                    |       C-e 行尾 |
|  M-a 句首                    |      M-e 句尾  |
|  M-< 文件头                 |     M-> 文件尾  |
|  M-g g 跳到某一行|  |

## 选择区域

C-@          标记

### 删除剪切复制粘贴

  C-d       		向后删除(delele)
  C-k 		删掉光标后至行尾
  M-w 		复制区域   
  C-w 		剪切/删除区域
  C-y 		粘贴
  M-y 		滚动选择粘贴内容

## 查找替换

  C-s 		向前查找 
  C-s C-r 		向后查找
  M-% 		替换

## 文件操作

  C-x C-f 		打开文件(find) 
  C-x C-s 		保存文件(save) 
  C-x C-w 	另存为(write)
  C-k        	关闭文件

### 窗口操作

  C-x b 		切换文件
  C-x 1 		关闭其它窗口
  C-x 2/C-x 3 	打开其它窗口
  C-x o 		跳到另一个窗口(other)

## 其它

  C-/ 		撤销
  M-$ 		拼写检查
  M-x 		输入命令
  C-g 		取消命令

# 






