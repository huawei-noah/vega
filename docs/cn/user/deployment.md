# 集群部署指导

## 1. 本地集群部署

### 1.1 部署前准备

本地集群部署Vega，需满足如下条件：

1. Ubuntu 18.04 or later (其他Linux发行版和版本未测试）。
2. CUDA 10.0
3. Python 3.7
4. pip3

集群在部署时，需要在每个集群节点中安装vega和一些必备的软件包，可执行如下命令进行安装：：

```bash
pip3 install noah-vega
```

```bash
python3 -m vega.tools.install_pkgs
```

除此之外，还需安装`MPI`软件，可参考附录[安装MPI](#MPI)完成安装过程。
注：若需要使用检测相关算法（SP-NAS），则还需要安装MMDetection， 可参考附录[安装MMDetection](#MMDetection)完成安装过程。

在各个主机上安装上述软件后，还需要[配置SSH互信](#ssh)，并[构建NFS](#nfs)。

以上前期工作完成后，请从Vega库中下载如下部署包[vega deploy package]()，部署包含有如下脚本，准备开始部署：

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

    在集群**主节点**中将`deploy_local_cluster.py`、`verify_local_cluster.py`、`deploy.yml`、`install_dependencies.sh`放到同一个文件夹中，执行如下命令，将Vega部署到主节点和从节点中：

    ```bash
    python3 deploy_local_cluster.py
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
    sudo mkdir -p /mnt/data
    ```

3. 挂载共享目录：

    ```bash
    sudo mount -t nfs 服务器ip:/data /mnt/data
    ```
