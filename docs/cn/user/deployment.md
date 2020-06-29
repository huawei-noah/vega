# 部署指导

## 1. 本地集群部署

### 1.1 部署前准备

本地集群部署Vega，需满足如下条件：

1. Ubuntu 16.04 or later (其他Linux发行版和版本未测试）。
2. CUDA 10.0 [下载](https://developer.nvidia.com/cuda-10.0-download-archive) [文档](https://docs.nvidia.com/cuda/archive/10.0/)
3. Python 3.7 [下载](https://www.python.org/downloads/release/python-376/)
4. 安装pip

5. 集群在部署时，需要预先安装一些必备的软件包，可下载脚本[install_dependencies.sh](../../../deploy/install_dependencies.sh)后安装：

```bash
bash install_dependencies.sh
```
6. 安装`MPI`软件, 可参考附录[安装MPI](#MPI)完成安装过程。
7. 安装 `MMDetection`(可选， 物体检测类算法所需的组件)， 可参考附录[安装MMDetection](#mmdetection)完成安装过程。

8. [配置SSH互信](#ssh)。 
9. [构建NFS](#nfs)。


以上准备工作完成后，请从Vega库中下载如下部署包[vega deploy package]()，部署包含有如下脚本，准备开始部署：

1. 部署脚本：`deploy_local_cluster.py`
2. 调测脚本：`verify_local_cluster.py`
3. 从节点启动脚本： `start_slave_worker.py`

### 1.2 部署

1. 首先配置部署信息到`deploy.yml`文件中，文件格式如下：

    ```yaml
    master: n.n.n.n     # master节点的IP地址
    listen_port: 8786   # 端口号
    slaves: ["n.n.n.n", "n.n.n.n", "n.n.n.n"]    # slave节点地址
    ```

2. 然后执行部署脚本

    在集群**主节点**中将`deploy_local_cluster.py`、`verify_local_cluster.py`、`vega-1.0.0.whl`、`deploy.yml`、`install_dependencies.sh`放到同一个文件夹中，执行如下命令，将Vega部署到主节点和从节点中：

    ```bash
    python deploy_local_cluster.py
    ```

    执行完成后，自动验证各个节点，会显示如下信息：

    ```text
    success.
    ```

## 参考

### <span id="mmdetection"> 安装MMDetection </span>

1. 下载MMDetection源码：

    在<https://github.com/open-mmlab/mmdetection>下载最新版本的MMDetection。

2. 安装：

    切换到mmdetection目录下，执行下述命令即可编译安装：

    ```bash
    sudo python3 setup.py develop
    ```

### <span id="MPI"> 安装MPI </span>

**安装MPI：**

1. 使用apt工具直接安装mpi

    ```bash
    sudo apt-get install mpi
    ```

2. 运行如下命令检查MPI是否可以运行

    ```bash
    mpirun
    ```

### 安装Apex

Apex需要从官网上获取最新的源码安装，不能直接使用pip库中的apex版本

1. 下载apex源码： 在<https://github.com/NVIDIA/apex>下载最新版本的apex。

2. 切换到apex目录下，执行下述命令即可编译安装：

    ```bash
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

### <span id="ssh"> 配置SSH互信 </span>

网络任意两台主机都需要支持SSH互信，配置方法为：

1. 安装ssh：
    `sudo apt-get install sshd`

2. 分别生成密钥：
    `ssh-keygen -t rsa` 会在~/.ssh/文件下生成id_rsa, id_rsa.pub两个文件，其中id_rsa.pub是公钥

3. 确认目录下的authorized_keys文件：
    若不存在需要创建， 并`chmod 600 ~/.ssh/authorized_keys`改变权限。

4. 拷贝公钥：
    分别将公钥id_rsa.pub内容拷贝到其他机器的authorized_keys文件中。

### <span id="nfs"> 构建NFS </span>

服务器端：

1. 安装NFS服务器：

    ```bash
    sudo apt install nfs-kernel-server
    ```

2. 编写配置文件，将共享路径写入配置文件中：

    ```bash
    sudo echo "/data *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
    ```

3. 创建共享目录：

    ```bash
    sudo mkdir -p /data
    ```

4. 重启nfs服务器：

    ```bash
    sudo service nfs-kernel-server restart
    ```

客户端：

1. 安装客户端工具：

    ```bash
    sudo apt install nfs-common
    ```

2. 创建本地挂载目录

    ```bash
    sudo mkdir -p /data
    ```

3. 挂载共享目录：

    ```bash
    sudo mount -t nfs 服务器ip:/data /data
    ```
**注意**：上述的共享目录(`/data`)的名字可以是任意的， 但需要保证主机和客户端的名字相同。
### CUDA安装指导

Ubuntu下cuda安装

1. 在英伟达官网下载安装包`cuda_10.0.130_410.48_linux.run`

2. 执行安装命令:
    命令如下：

    ```bash
    sudo sh cuda_10.0.130_410.48_linux.run
    ```

    在执行过程中，会有一系列提示，选择默认设置即可。需要注意的是其中有个选择，询问是否安装`NVIDIA Accelerated Graphics Driver`：
    `Install NVIDIA Accelerated Graphics Driver for Linux‐x86_64?`
    请选择 `no`

3. 环境变量配置：
    执行：

    ```bash
    sudo gedit /etc/profile
    ```

    在profile文件的最后面添加内容：

    ```bash
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ```

    保存profile文件，并执行以下命令，使环境变量立即生效

    ```bash
    source /etc/profile
    ```

4. 安装cuda sample:
    进入/usr/local/cuda/samples, 执行下列命令来build samples：

    ```bash
    sudo make all -j8
    ```

    全部编译完成后， 进入/usr/local/cuda/samples/1_Utilities/deviceQuery, 运行deviceQuery:

    ```bash
    ./deviceQuery
    ```
