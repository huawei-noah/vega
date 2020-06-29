# 安装

## 1. 安装前的准备

安装Vega的主机有GPU，且需要满足如下要求：

1. Ubuntu 16.04 or later (其他Linux发行版未完全测试）。
2. CUDA 10.0 [下载](https://developer.nvidia.com/cuda-10.0-download-archive) [文档](https://docs.nvidia.com/cuda/archive/10.0/)
3. Python 3.7 [下载](https://www.python.org/downloads/release/python-376/)
4. pip

在安装Vega前，除了通过pip安装一些必备的软件包外，还需要单独安装 MMDetection 和 pycocotools。

### 1.1 通过pip安装必备软件包

在安装前，需要预先安装一些必备的软件包，可下载脚本[install_dependencies.sh](../../../deploy/install_dependencies.sh)后安装：

```bash
bash install_dependencies.sh
```

在安装完成后，需要执行如下命令，确保环境设置正确：

```bash
which dask-scheduler
```

若该命令返回dask-scheduler的文件位置，则安装成功。若未返回路径信息，可考虑重新登录服务器，使得安装脚本设置的路径生效。
若该命令还未返回dask-scheduler文件位置，那需要将路径`$HOME/.local/bin/`配置到PATH环境变量中。

### 1.2 安装MMDetection

首先下载[mmdetection-1.0rc1.zip](https://github.com/open-mmlab/mmdetection/archive/v1.0rc1.zip)。

然后解压后安装：

```bash
unzip mmdetection-1.0rc1.zip
cd mmdetection-1.0rc1
python3 setup.py develop --user
```

## 2. 安装Vega

完成以上准备后，下一步是在`release`下载`vega-0.9.1-py3-none-any.whl`，执行`pip`安装：

```bash
pip3 install vega-0.9.1.py3-none-any.whl
```

安装完成后，可以尝试在python中引入vega库，确保安装成功：

```text
$ python3
Python 3.7.6 (default, Feb 27 2020, 19:54:18)
[GCC 5.3.1 20160413] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import vega
>>>
```

## FAQ

1. 出现异常信息 `ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory` 解决方法。

    Vega使用了三方软件opencv-python，该软件对操作系统的库有依赖，一般情况下该软件安装完成后可以正常使用，但在某些场景下，会出现如上错误提示，这时需要执行如下命令，解决该问题：

    ```bash
    sudo apt install libglib2.0-0
    ```

2. 若需要支持混合精度和SyncBN，则需要安装Apex，下载Apex源码[apex-master.zip](https://codeload.github.com/NVIDIA/apex/zip/master)，执行如下命令安装：

    ```bash
    unzip apex-master.zip
    cd apex-master
    pip3 install --user -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
