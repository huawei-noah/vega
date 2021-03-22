# 集群部署指导

## 1. 本地集群部署

### 1.1 部署

本地集群部署Vega，需满足如下条件：

1. Ubuntu 18.04 or later。
2. CUDA 10.0
3. Python 3.7
4. pip3

**注： 若需要在Ascend 910集群上部署，请和我们联系。**

集群在部署时，需要在每个集群节点中安装vega和一些必备的软件包，可执行如下命令进行安装：

```bash
pip3 install --user --upgrade noah-vega
```

除此之外，还需安装`MPI`软件，可参考附录[安装MPI](#MPI)完成安装过程。

在各个主机上安装上述软件后，还需要[配置SSH互信](#ssh)，并[构建NFS](#nfs)。

以上工作完成后，集群已部署完成。

### 1.2 校验

集群部署完成后，请执行以下命令检查集群是否可用：

```bash
vega-verify-cluster -m <master IP> -s <slave IP 1> <slave IP 2> ... -n <shared NFS folder>
```

例如：

```bash
vega-verify-cluster -m 192.168.0.2 -s 192.168.0.3 192.168.0.4 -n /home/alan/nfs_folder
```

校验结束后，会有显示"All cluster check items have passed."。
若校验中出现错误，请根据异常信息调整集群。

## 参考

### <span id="MPI"> 安装MPI </span>

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

NFS服务器设置：

1. 安装NFS服务器：

    ```bash
    sudo apt install nfs-kernel-server
    ```

2. 创建NFS服务器共享目录，如下例中的 `/<user home path>/nfs_cache`：

    ```bash
    cd ~
    mkdir nfs_cache
    ```

3. 将共享目录写入配置文件`/etc/exports`中：

    ```bash
    sudo bash -c "echo '/home/<user home path>/nfs_cache *(rw,sync,no_subtree_check,no_root_squash,all_squash)' >> /etc/exports"
    ```

4. 将共享目录设置为`nobody`用户

    ```bash
    sudo chown -R nobody: /<user home path>/nfs_cache
    ```

5. 重启nfs服务器：

    ```bash
    sudo service nfs-kernel-server restart
    ```

在每个服务器都需要配置NFS客户端：

1. 安装客户端工具：

    ```bash
    sudo apt install nfs-common
    ```

2. 创建本地挂载目录

    ```bash
    cd ~
    mkdir -p ./nfs_folder
    ```

3. 挂载共享目录：

    ```bash
    sudo mount -t nfs <服务器ip>:/<user home path>/nfs_cache /<user home path>/nfs_folder
    ```

挂载完成后，`/<user home path>/nfs_folder` 就是多机集群的工作目录，请在该目录下运行Vega程序。
