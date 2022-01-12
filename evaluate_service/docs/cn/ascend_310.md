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

若`/usr/local/Ascend/`已存在，则需要在安装前需要检查是否已安装了较旧的Driver和CANN包，
请使用如下命令查询各个组件的版本号：

```bash
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/nnae/latest/ascend_nnae_install.info
cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/nnrt/latest/arm64-linux/ascend_nnrt_install.info
cat /usr/local/Ascend/tfplugin/latest/ascend_tfplugin_install.info
```

若版本号较低，需要使用root账号执行卸载：

```bash
/usr/local/Ascend/driver/script/uninstall.sh
/usr/local/Ascend/nnae/latest/script/uninstall.sh
/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/script/uninstall.sh
/usr/local/Ascend/nnrt/latest/arm64-linux/script/uninstall.sh
/usr/local/Ascend/tfplugin/latest/script/uninstall.sh
```

若使用X86平台，请将如上命令中包含的目录中的`arm64-linux`替换为`x86_64-linux`。

若nnae、ascend-toolkit、nnrt、tfplugin使用非root安装，请使用该用户卸载。

## 2 安装Driver和CANN

使用root用户执行如下命令安装，如下版本号供参考：

```bash
chmod +x *.run
./A300-3000-3010-npu-driver_21.0.2_linux-aarch64.run --full
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

## 3 设置环境变量

请设置如下环境变量：

```bash
export ASCEND_HOME=/usr/local/Ascend
export HOME_DIR=/home/<username>
export PATH=$HOME_DIR/.local/bin:$PATH
source /usr/local/Ascend/nnae/set_env.sh
source /usr/local/Ascend/nnrt/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/atc/lib64
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$PYTHONPATH
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME_DIR/.local/lib/python3.7/site-packages/evaluate_service/security:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME_DIR/.local/lib/python3.7/site-packages/evaluate_service/security/kmc/aarch64:$LD_LIBRARY_PATH
```

其中`<username>`为用户目录，`$NPU_HOST_LIB`为`libascendcl.so`的路径, 需要根据`libascendcl.so`实际所在的位置配置此变量。
