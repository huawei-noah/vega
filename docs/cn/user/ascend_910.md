# 部署Ascend环境

请参考Ascend官方文档部署Ascend环境，如下安装指导是安装过程中的关键步骤，若安装过程中出现问题，请以官方文档为准。
在进行部署前，请在官方网站下载安装包。

## 1 检查已安装的Driver和CANN版本

若是全新的Ascend主机，需要检查是否存在`/usr/local/HiAi`目录，若存在，需要使用root账号执行如下命令卸载该目录：

```bash
/usr/local/HiAi/uninstall.sh
```

需要使用非root账号执行如下命令创建`Ascend`目录，并给该目录设置为用户`HwHiAiUser`可访问：

```bash
mkdir /usr/local/Ascend/
sudo chown -R :HwHiAiUser /usr/local/Ascend/
sudo chmod -R 750 /usr/local/Ascend/
```

若`/usr/local/Ascend/`已存在，则需要在安装前需要检查是否已安装了较旧的Driver和CANN包，请使用如下命令查询各个组件的版本号：

```bash
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/firmware/version.info
cat /usr/local/Ascend/nnae/latest/ascend_nnae_install.info
cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/tfplugin/latest/ascend_tfplugin_install.info
```

如上`/usr/local/Ascend`目录是较常使用的目录，也可能是``

若版本号较低，需要使用root账号执行卸载：

```bash
/usr/local/Ascend/driver/script/uninstall.sh
/usr/local/Ascend/firmware/script/uninstall.sh
/usr/local/Ascend/nnae/latest/script/uninstall.sh
/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/script/uninstall.sh
/usr/local/Ascend/tfplugin/latest/script/uninstall.sh
```

若nnae、ascend-toolkit、tfplugin使用非root安装，请使用该用户卸载。

## 2 安装Driver和CANN

使用root用户执行如下命令安装，如下版本号供参考：

```bash
chmod +x *.run
./A800-9000-npu-driver_21.0.3.1_linux-aarch64.run --full
./A800-9000-npu-firmware_1.79.22.4.220.run --full
```

执行如下命令，确认安装是否成功：

```bash
npu-smi info
```

使用非root用户安装其他包，在安装前，需要将该用户设置为和`HwHiAiUser`同组：

```bash
usermod -a -G HwHiAiUser <username>
```

```bash
./Ascend-cann-nnae_5.0.T306_linux-aarch64.run --install
./Ascend-cann-nnrt_5.0.T306_linux-aarch64.run --install
./Ascend-cann-tfplugin_5.0.T306_linux-aarch64.run --install
./Ascend-cann-toolkit_5.0.T306_linux-aarch64.run --install
```

安装完成后，根据提示需要重启主机。

## 3 配置rank_table_file

请参考Ascend的官方文档，执行`hccn_tool`命令，生成`rank_table_file`。

## 4 配置环境变量

需要配置如下环境变量，建议放入`~/.bashrc`中：

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

如上`<username>`为当前用户名，`<job_id>`请设置一个整数，如`10087`，`<rank_table_file.json>`请设置为该文件的全路径。

## 5 安装Vega及依赖包

先升级pip到最新版本：

```bash
pip3 install --user --upgrade pip
```

先安装nnae、topi、hccl等组件包：

```bash
export fwk_path='/usr/local/Ascend/nnae/latest'
export te_path=${fwk_path}'/fwkacllib/lib64/te-*.whl'
export topi_path=${fwk_path}'/fwkacllib/lib64/topi-*.whl'
export hccl_path=${fwk_path}'/fwkacllib/lib64/hccl-*.whl'
pip3 install --user ${te_path}
pip3 install --user ${topi_path}
pip3 install --user ${hccl_path}
```

再安装noah-vega，因Ascend环境特殊性，注意不要安装依赖包：

```bash
pip3 install --user --no-deps noah-vega
```

再通过如下的命令查看Vega的依赖包：

```bash
pip3 show noah-vega
```

另外要注意的是，dask和distributed这两个包，需要安装如下版本：

```bash
pip3 install --user distributed==2021.7.0
pip3 install --user dask==2021.7.0
```
