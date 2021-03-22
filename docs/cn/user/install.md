# 安装

## 1. 安装前的准备

安装Vega的主机有GPU，且需要满足如下要求：

1. Ubuntu 18.04 or EulerOS 2.0 SP8
2. CUDA 10.0 or CANN 20.1
3. Python 3.7
4. pip3

## 2. 安装Vega

执行如下命令安装：

```bash
pip3 install --user --upgrade noah-vega
```

**重要：**

1. 在安装vega前，请先执行如下命令升级pip：

    ```bash
    pip3 install --user --upgrade pip
    ```

2. 安装完成后，若`~/.local/bin`不在`$PATH`环境变量中，则需要重新登录，使环境变量生效。

若是Ascend 910训练环境，请联系我们。

## 3. 支持混合精度和SyncBNN

需要支持混合精度和SyncBN，则需要安装Apex，下载Apex源码 `apex-master.zip` ，请根据该软件主页上的安装说明来安装该软件。
