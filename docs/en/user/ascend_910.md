# Deploy the Ascend environment.

Deploy the Ascend environment by referring to the Ascend official document. The following 
installation guide is a key step during the installation. If an error occurs during the 
installation, refer to the official document.
Before the deployment, download the installation package from the official website.

## 1 Check the install Driver and CANN Versions

For a new Ascend host, check whether the `/usr/local/HiAi` directory exists. If yes, 
run the following command as root user to uninstall the directory:

```bash
/usr/local/HiAi/uninstall.sh
```

Run the following commands as a non-root user to create the `Ascend` directory 
and make the directory accessible to the `HwHiAiUser` user:

```bash
mkdir /usr/local/Ascend/
sudo chown -R :HwHiAiUser /usr/local/Ascend/
sudo chmod -R 750 /usr/local/Ascend/
```

If `/usr/local/Ascend/` exists, check whether the old Driver and CANN packages have been 
installed before the installation. Run the following command to query the version number of 
each component:

```bash
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/firmware/version.info
cat /usr/local/Ascend/nnae/latest/ascend_nnae_install.info
cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/tfplugin/latest/ascend_tfplugin_install.info
```

If the version is older than expected, uninstall it as root user.

```bash
/usr/local/Ascend/driver/script/uninstall.sh
/usr/local/Ascend/firmware/script/uninstall.sh
/usr/local/Ascend/nnae/latest/script/uninstall.sh
/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/script/uninstall.sh
/usr/local/Ascend/tfplugin/latest/script/uninstall.sh
```

If nnae, ascend-toolkit, and tfplugin are not installed by the root user, uninstall them as the user.

## 2 Install the driver and CANN

Run the following command as the root user to install the software. The following version 
is for reference only:

```bash
chmod +x *.run
./A800-9000-npu-driver_21.0.3.1_linux-aarch64.run --full
./A800-9000-npu-firmware_1.79.22.4.220.run --full
```

Run the following command to check whether the installation is successful:

```bash
npu-smi info
```

Before installing other packages as a non-root user, set this user to the same group as `HwHiAiUser`.

```bash
usermod -a -G HwHiAiUser <username>
```

```bash
./Ascend-cann-nnae_5.0.T306_linux-aarch64.run --install
./Ascend-cann-nnrt_5.0.T306_linux-aarch64.run --install
./Ascend-cann-tfplugin_5.0.T306_linux-aarch64.run --install
./Ascend-cann-toolkit_5.0.T306_linux-aarch64.run --install
```

After the installation is completed, restart the host as prompted.

## 3 Configure rank_table_file.

Run the `hccn_tool` command to generate `rank_table_file` by referring to the official Ascend document.

## 4 Configure environment Variables

The following environment variables need to be configured. 
You are advised to place them in the `~/.bashrc` directory:

```bash
export HOME_DIR=/home/<username>
export HOST_ASCEND_BASE=/usr/local/Ascend
export JOB_ID=<job_id>
export DEVICE_ID=0
export RANK_TABLE_FILE=<rank_table_file.json>
export RANK_ID=0
export RANK_SIZE=8
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_TASK_INDEX=0
export TF_CPP_MIN_LOG_LEVEL=3
export LD_PRELOAD=export LD_PRELOAD=/lib64/libgomp.so.1:$HOME_DIR/.local/lib/python3.7/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export GLOG_v=3
export USE_NPU=True
source /usr/local/Ascend/tfplugin/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnae/set_env.sh
export PATH=$HOME_DIR/.local/bin:$PATH
export PYTHONPATH=$HOME_DIR/.local/lib/python3.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME_DIR/.local/lib/python3.7/site-packages/vega/security/kmc/aarch64:$LD_LIBRARY_PATH
```

In the preceding command, `<username>` is the current user name. `<job_id>` must be an integer, 
for example, `10087`. `<rank_table_file.json>` must be the full path of the file.

## 5 Install Vega and Dependency Packages

Upgrade the PIP to the latest version.

```bash
pip3 install --user --upgrade pip
```

Install the nnae, topi, and hccl component packages.

```bash
export fwk_path=' /usr/local/Ascend/nnae/latest'
export te_path=${fwk_path}'/fwkacllib/lib64/te-*.whl'
export topi_path=${fwk_path} '/fwkacllib/lib64/topi-*.whl'
export hccl_path=${fwk_path} '/fwkacllib/lib64/hccl-*.whl'
pip3 install --user ${te_path}
pip3 install --user ${topi_path}
pip3 install --user ${hccl_path}
```

Install noah-vega. Do not install the dependency package because of the special environment of Ascend.

```bash
pip3 install --user --no-deps noah-vega
```

Run the following command to view the Vega dependency package:

```bash
pip3 show noah-vega
```

Note that the following versions must be installed for the dask and distributed packages:

```bash
pip3 install --user distributed==2021.7.0
pip3 install --user dask==2021.7.0
```