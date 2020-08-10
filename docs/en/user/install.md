# Installation

## 1. Preparations Before Installation

The host where the Vega is installed has a GPU and meets the following requirements:

1. Ubuntu 16.04 or later (other Linux distributions are not fully tested)
2. CUDA 10.0 [download](https://developer.nvidia.com/cuda-10.0-download-archive) [ref](https://docs.nvidia.com/cuda/archive/10.0/)
3. Python 3.7 [download](https://www.python.org/downloads/release/python-376/)
4. pip

Before installing Vega, you need to install MMDetection and pycocotools in addition to some mandatory software packages through pip.

### 1.1 Installing Mandatory Software Packages Using pip

Before deploying a cluster, you need to install some mandatory software packages. You can download the script [install_dependencies.sh](../../../deploy/install_dependencies.sh) and install them.

```bash
bash install_dependencies.sh
```

After installation, you need to execute the following command to ensure that the environment is set correctly:

```bash
which dask-scheduler
```

If the command returns the file location of dask-scheduler, the installation is successful. If the path information is not returned, consider logging in to the server again to make the path set by the installation script take effect.
If the command has not returned the dask-scheduler file location, then you need to configure the path `$HOME/.local/bin/` into the PATH environment variable.

### 1.2 Installing the MMDetection

Download [mmdetection-1.0rc1.zip](https://github.com/open-mmlab/mmdetection/archive/v1.0rc1.zip)。

Decompress the downloaded package and install it.

```bash
unzip mmdetection-1.0rc1.zip
cd mmdetection-1.0rc1
python3 setup.py develop --user
```

## 2. Install Vega

After the preceding preparations are complete, download the vega-1.0.0-py3-none-any.whl file in the release directory and run the pip command to install it.

```bash
pip3 install vega-1.0.0.py3-none-any.whl
```

After the installation is complete, import the VEGA library to Python to ensure that the installation is successful.

```text
$ python3
Python 3.7.6 (default, Feb 27 2020, 19:54:18)
[GCC 5.3.1 20160413] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import vega
>>>
```

## FAQ

1. An exception message appears `ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory` solution.

     Vega uses the three-party software opencv-python, which depends on the operating system library. In general, the software can be used normally after the installation is completed, but in some scenarios, the above error message will appear, then you need to execute the following command To solve the problem:

     ```bash
     sudo apt install libglib2.0-0
     ```

2. If you need to support mixed precision and SyncBN, you need to install Apex, download the Apex source code [apex-master.zip](https://codeload.github.com/NVIDIA/apex/zip/master), execute the following command to install:

    ```bash
    unzip apex-master.zip
    cd apex-master
    pip3 install --user -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
