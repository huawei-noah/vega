# Deployment Guide

## 1. Local cluster deployment

### 1.1 Deployment

The following conditions must be met when the Vega is deployed in a local cluster:

1. Ubuntu 18.04 or later
2. CUDA 10.0
3. Python 3.7
4. pip

**Note: If you need to deploy the Ascend 910 cluster, contact us.**

During cluster deployment, you need to install the Vega and some mandatory software packages on each cluster node by running the following commands:

```bash
pip3 install --user --upgrade noah-vega
```

In addition, you need to install the `MPI' software. For details, see [Installing the MPI](#MPI).

After installing the preceding software on each host, you need to configure SSH mutual trust (#ssh) and build NFS (#nfs).

After the preceding operations are complete, the cluster has been deployed.

### 1.2 Verify

After the cluster is deployed, run the following command to check whether the cluster is available:

```bash
vega-verify-cluster -m <master IP> -s <slave IP 1> <slave IP 2> ... -n <shared NFS folder>
```

For example:

```bash
vega-verify-cluster -m 192.168.0.2 -s 192.168.0.3 192.168.0.4 -n /home/alan/nfs_folder
```

After the verification is complete, the message "All cluster check items have passed." is displayed.
If an error occurs during the verification, please adjust the cluster based on the exception information.

## Reference

### <span id="MPI"> Install MPI</span>

1. Use the apt tool to install MPI directly

    ```bash
    sudo apt-get install mpi
    ```

2. Run the following commandes to check mpi is working.

    ```bash
    mpirun
    ```

### <span id="ssh"> Configure SSH mutual trust </span>

Any two hosts on the network must support SSH mutual trust. The configuration method is as follows:

1. Install SSH.
    `sudo apt-get install sshd`

2. Indicates the public key.
    `ssh-keygen -t rsa` two file id_rsa, id_rsa.pub will be create in folder ~/.ssh/, id_rsa.pub is public key.

3. Check the authorized_keys file in the directory. If the file does not exist, create it and run the chmod 600 ~/.ssh/authorized_keys command to change the permission.

4. Copy the public key id_rsa.pub to the authorized_keys file on other servers.

### <span id="nfs"> Building NFS </span>

NFS server settings:

1. Install the NFS server.

    ```bash
    sudo apt install nfs-kernel-server
    ```

2. Create a shared directory on the NFS server, for example, `/<user home path>/nfs_cache`.

    ```bash
    cd ~
    mkdir nfs_cache
    ```

3. Write the shared directory to the configuration file `/etc/exports`.

    ```bash
    sudo bash -c "echo '/home/<user home path>/nfs_cache *(rw,sync,no_subtree_check,no_root_squash,all_squash)' >> /etc/exports"
    ```

4. Set the shared directory to the `nobody` user.

    ```bash
    sudo chown -R nobody: /<user home path>/nfs_cache
    ```

5. Restart the NFS server.

    ```bash
    sudo service nfs-kernel-server restart
    ```

The NFS client must be configured on each server.

1. Install the client tool.

    ```bash
    sudo apt install nfs-common
    ```

2. Create a local mount directory.

    ```bash
    cd -
    mkdir -p ./nfs_folder
    ```

3. Mount the shared directory.

    ```bash
    sudo mount -t nfs < Server ip>:/<user home path>/nfs_cache /<user home path>/nfs_folder
    ```

After the mounting is complete, `/<user home path>/nfs_folder` is the working directory of the multi-node cluster. Run the Vega program in this directory.
