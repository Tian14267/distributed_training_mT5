# Distributed MT5



## 主要内容

本算法是一款面向MT5大模型的多卡prompt训练方案。

* 该大模型的prompt训练支持CPU、多卡GPU训练。
* 提供训练代码和测试代码



## 配置环境


- 安装 Conda:  https://docs.conda.io/en/latest/miniconda.html
- 创建 Conda 环境:

``` sh
conda create -n mt5 python=3.10
conda activate mt5
pip install -r requirements.txt
```
- 安装多卡训练环境。这里使用的多卡训练工具为 horovod:
```
安装 NCCL
安装 horovod：0.27.0
    安装方法：HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
https://github.com/horovod/horovod/blob/master/docs/pytorch.rst
```


## 训练模型

1：准备数据
    可以参照文件夹 data 中的数据进行相应的数据准备。

2：模型训练：
```
这里以3张卡为例子：
CUDA_VISIBLE_DEVICES="1,2,3" horovodrun -np 3 python mT5_train.py
```

3：模型测试
详见测试代码 `mt5_test.py`

注：相应的CPU训练方案，见`mT5_train_cpu.py`。但是不推荐实验CPU。

4: 同时也提供Lora训练方案
```
sh run_train_mT5_Lora.sh
```

### 新手常见问题

基本都是环境搭建问题。可以自行解决。




