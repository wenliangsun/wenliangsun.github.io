---
layout: post
title: "mmdetection 框架中踩过的坑(DOTA 数据集)"
subtitle: 'Object Detection structure bugs'
author: "WenlSun"
header-style: text
tags:
  - Object Detection
  - 笔记
  - pytorch
---

### 编译出错 

+ `bash: ./complie.sh:Permission denied`
+ `/usr/bin/env: ‘bash\r’: No such file or directory`
+ 解决方法如下图所示

![](/img/post-mmdetection-bug1.png)

+ 同样的操作处理`dist_train.sh` 如果需要指定使用那一块`GPU`需要修改`dist_train.sh`文件为：

  ```
  #!/usr/bin/env bash
  
  PYTHON=${PYTHON:-"python"}
  
  CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
  ```


### 每次修改`Python`文件都需要重新编译问题

在安装使用`python setup.py develop or pip install -e  `来解决。

![](/img/post-mmdetection-bug2.png)

### 运行 dist_train.sh 文件报 Address already in use错误

原因是一台机子上跑了两个mmdetection代码导致节点冲突，解决办法有两个：

+ 修改配置文件中的`dist_params = dict(backend='nccl')`为`dist_params = dict(backend='nccl', master_port=xxxx)`其中 xxxx 是指没有使用的任意端口。

+ 修改`dist_params.sh`为：

  ```
  CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.launch --nproc_per_node=$2 --master_port xxxx $(dirname "$0")/train.py $1 --launcher pytorch ${@:3} #其中 xxxx 是指没有使用的任意端口
  ```



bug还在挖掘中。。。