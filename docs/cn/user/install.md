# 安装

## 1. 安装前的准备

安装Vega的主机有GPU，且需要满足如下要求：

1. Ubuntu 18.04 or later (其他Linux发行版未完全测试）。
2. CUDA 10.0
3. Python 3.7
4. pip3

## 2. 安装Vega

执行如下命令安装：

```bash
pip3 install --user noah-vega --upgrade pip
```

Vega安装过程中并不安装所有的依赖包，在完成Vega安装后，还需要执行如下命令安装依赖的软件包：

```bash
python3 -m vega.tools.install_pkgs
```

## 3. 安装MMDetection

若需要使用检测相关算法（SP-NAS），则还需要安装MMDetection，下载 `mmdetection-1.0rc1.zip`，请根据该软件主页上的安装说明来安装该软件。

## 4. 支持混合精度和SyncBNN

需要支持混合精度和SyncBN，则需要安装Apex，下载Apex源码 `apex-master.zip` ，请根据该软件主页上的安装说明来安装该软件。
