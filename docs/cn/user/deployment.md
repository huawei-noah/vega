# 集群部署指导

## 1. 本地集群部署

本地集群部署Vega，需满足如下条件：

1. Ubuntu 18.04 or EulerOS 2.0 SP8
2. CUDA 10.0 or CANN 20.1
3. Python 3.7 or later
4. pytorch, tensorflow(>1.14, <2.0) or mindspore

集群在部署时，需要在每个集群节点中安装vega：

```bash
pip3 install --user --upgrade noah-vega
```

除此之外，还需安装`MPI`软件，可参考附录[安装MPI](#MPI)完成安装过程。

在各个主机上安装上述软件后，还需要[配置SSH互信](#ssh)，并[构建NFS](#nfs)。

以上工作完成后，集群已部署完成。

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

NFS是集群中用于数据共享的常用系统，若你所使用的集群中已经有NFS系统，请直接使用已有的NFS系统。

以下配置NFS的简单指导，可能不适用于所有的NFS系统，请根据实际集群环境调整。

在配置NFS服务器前，需要确定当前用户的在集群中的各个主机上的UID是否是同样的数值。若UID不相同，会造成无法访问NFS共享目录，需要调整当前用户的UID为同一个数值，同时要避免和其他用户的UID冲突。

查询当前用户的UID：

```bash
id <user name>
```

修改当前用的UID，（请慎重修改，请咨询集群系统管理员获取帮助）：

```bash
sudo usermod <user name> -u <new UID>
```

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

4. 重启nfs服务器：

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
