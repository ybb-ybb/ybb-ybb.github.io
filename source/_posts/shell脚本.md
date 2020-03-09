---
date: 2020-02-10 13:32:00
description: shell脚本最基本的使用
title: shell脚本
author: 鱼摆摆
comments: true
tags: 
 - shell
photos: https://cdn.jsdelivr.net/gh/ybb-ybb/CDNrepository@1.1/img/fig.jpg
categories: 瞎折腾
---
```shell
# 指定解释器
#！/bin/bash
# 向窗口输出文本
echo "Hello world!"
printf "Hello world!"
# for循环示例,使用变量要加$符号
for file in `ls /etc`
do
	echo "${file}"
done
# 双引号和单引号
# 双引号中可以有变量，单引号中的变量是无效的
# if-else语句
if condition1
then
	command1
elif condition2
then
	commed2
else
	command3
fi
```

