# Evaluate Service

## 1. 简介

模型评估服务是用于评估模型在特定硬件设备上的性能，如评估剪枝和量化后的模型在Atalas 200 DK上的准确率、模型大小和时延等。
评估服务目前支持的硬件设备为Davinci推理芯片（Atalas200 DK、ATLAS300产品和开发板环境Evb)和手机，后继会扩展支持更多的设备。

评估服务为CS架构， 评估服务在服务端部署， 客户端通过`REST`接口向服务端发送评估请求和获取结果。Vega在进行网络架构搜索时，可以利用评估服务进行实时检测模型性能。在搜索阶段产生备选网络后，可以将该网络模型发送给评估服务，评估服务完成模型评估后，返回评估结果给Vega，Vega根据评估结果，进行后继的搜索。这种实时的在实际的设备上的评估，有利于搜索出对实际硬件更加友好的网络结构。

## 2. 规格

支持的模型和硬件设备

| 算法 | 模型 | Atalas 200 DK | Bolt |
| :--: | :--: | :--: | :--: |
| Prune-EA | PruneResNet | 支持 | coming soon |
| Quant-EA | ResNet-Quant | | |
| ESR-EA | ESRN | | coming soon |
| CycleSR | CycleSRModel | | | |
| CARS | CARSDartsNetwork | | |
| Adlaide-EA | AdelaideFastNAS | | |
| Auto-Lane | ResNetVariantDet | | |
| Auto-Lane | ResNextVariantDet | | |
| JDD | JDDNet |
| SM-NAS | ResNet_Variant |
| SM-NAS | ResNet_Variant |
| SP-NAS | spnet_fpn |
| SR-EA | MtMSR | | coming soon |

## 3. 评估服务部署

该教程为服务端的部署， 客户端无须部署。环境安装步骤为可选步骤，如果环境已经安装配置， 或者不需要使用相应环境， 则可直接跳过相应环境的安装配置步骤。

### 3.1 环境安装配置

### 3.1.1 安装 Atalas200 DK 环境（可选）

#### 3.1.1.1 准备工作

1. 准备好一张8GB以上SD卡及读卡器。
2. 已安装 ubuntu 16.04.3 的服务器一台，这台服务器就是评估服务器。
3. 下载系统镜像： [ubuntu-16.04.3-server-arm64.iso](http://old-releases.ubuntu.com/releases/16.04.3/ubuntu-16.04.3-server-arm64.iso)
4. 下载制卡脚本： make_sd_card.py 和 make_ubuntu_sd.sh，下载地址：<https://github.com/Ascend-Huawei/tools>
5. 下载开发者运行包： mini_developerkit-1.3.T34.B891.rar  下载地址：<https://www.huaweicloud.com/ascend/resources/Tools/1>

#### 3.1.1.2 安装和配置Atalas200 DK

1. 将SD卡放入读卡器，并将读卡器与Ubuntu服务器的USB接口连接。
2. Ubuntu服务器上安装依赖项：

    ```bash
    apt-get install qemu-user-static binfmt-support python3-yaml gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    ```

3. 执行如下命令查找SD卡所在的USB设备名称。

    ```bash
    fdisk -l
    ```

4. 运行SD制卡脚本开始制卡，此处“USB设备名称”即为上一步得到的名称。

    ```bash
    python3 make_sd_card.py  local  USB设备名称
    ```

5. 制卡成功后，将SD卡从读卡器取出并插入Atlas 200 DK开发者板卡槽, 上电Atlas 200 DK开发者板。

#### 3.1.1.3 安装和配置评估服务器环境

1. 下载安装DDK包及同步lib库

    下载地址：<https://www.huaweicloud.com/ascend/resources/Tools/1>
    安装步骤可参考官方文档： <https://www.huaweicloud.com/ascend/doc/atlas200dk/1.32.0.0(beta)/zh/zh-cn_topic_0204690657.html>

2. 配置交叉编译环境
    需要在评估服务器上安装Atlas200 DK所需的编译环境，执行如下命令：

    ```bash
    sudo apt-get install g++-aarch64-linux-gnu
    ```

3. 在服务器的 `/etc/profile` 中配置如下环境变量：

    ```bash
    export DDK_PATH=/home/ly/huawei/ddk
    export PYTHONPATH=$DDK_PATH/site-packages/te-0.4.0.egg:$DDK_PATH/site-packages/topi-0.4.0.egg
    export LD_LIBRARY_PATH=$DDK_PATH/uihost/lib:$DDK_PATH/lib/x86_64-linux-gcc5.4
    export PATH=$PATH:$DDK_PATH/toolchains/ccec-linux/bin:$DDK_PATH/uihost/bin
    export TVM_AICPU_LIBRARY_PATH=$DDK_PATH/uihost/lib/:$DDK_PATH/uihost/toolchains/ccec-linux/aicpu_lib
    export TVM_AICPU_INCLUDE_PATH=$DDK_PATH/include/inc/tensor_engine
    export TVM_AICPU_OS_SYSROOT=/home/ly/tools/sysroot/aarch64_Ubuntu16.04.3
    export NPU_HOST_LIB=/home/ly/tools/1.32.0.B080/RC/host-aarch64_Ubuntu16.04.3/lib
    export NPU_DEV_LIB=/home/ly/tools/1.32.0.B080/RC/host-aarch64_Ubuntu16.04.3/lib
    ```

4. 配置ssh互信
    由于评估服务器和Atlas200 DK 之间需要进行文件传输以及远端命令的执行，因此需要分别在两个环境上配置ssh互信，确保脚本能够自动化运行。

    a. 安装ssh：`sudo apt-get install ssh`  
    b. 生成密钥：`ssh-keygen -t rsa` 会在~/.ssh/文件下生成id_rsa, id_rsa.pub两个文件，其中id_rsa.pub是公钥  
    c. 确认目录下的authorized_keys文件。若不存在需要创建， 并`chmod 600 ~/.ssh/authorized_keys`改变权限。  
    d. 拷贝公钥：分别将公钥id_rsa.pub内容拷贝到其他机器的authorized_keys文件中。  
    **注意**： 以上步骤需要在评估服务器和Atlas 200 DK 分别执行一遍， 确保这两台机器之间ssh互信。

### 3.1.2 安装配置Atlas300环境(可选)

参考华为图灵官方教程： <https://support.huawei.com/enterprise/zh/ai-computing-platform/a300-3000-pid-250702915>

### 3.1.3 安装和配置手机环境(可选)

#### 3.1.3.1 准备工作

1. 准备Kirin 980手机1台，推荐Nova 5。
2. 已安装 ubuntu 16.04.3 的服务器一台，这台服务器就是评估服务器，可以和Atalas 200 DK 评估服务器共用一台服务器。

#### 3.1.3.2 安装和配置评估服务器和手机

1. 在linux 系统服务器上安装adb工具。

    ```bash
    apt install adb
    ```

2. 通过USB端口将手机接入到评估服务器，并打开开发者选项（TODO: 详细说明），并在评估服务器上执行如下命令：

    ```bash
    adb devices
    ```

    出现如下信息即为连接成功：

    ```text
    List of devices attached
    E5B0119506000260 device
    ```

#### 3.1.3.3 设备连接失败的处理

若在服务器上通过 `adb devices` 命令不能获取到设备，则可以通过以下步骤尝试连接：

1. 在评估服务器上执行`lsusb`命令， 出现设备列表， 找到设备的ID。

2. 编辑51-android.rules 文件:

    ```bash
    sudo vim /etc/udev/rules.d/51-android.rules
    ```

    写入如下内容

    ```text
    SUBSYSTEM=="usb", ATTR{idVendor}=="12d1", ATTR{idProduct}=="107e", MODE="0666"
    ```

    注意： 上面的12d1和107e是上一步查询到的ID。

3. 编辑adb_usb.ini 文件:

    ```bash
    vim ~/.android/adb_usb.ini
    ```

    写入如下内容：

    ```text
    0x12d1
    ```

    注意： 上面的12d1是步骤5.1查询到的ID。

4. 重启adb服务

    ```bash
    sudo adb kill-server
    sudo adb start-server
    ```

5. 再次执行`adb devices`，确认是否连接成功。

### 3.2 安装和启动评估服务

下载评估服务的代码[evaluate_service](../../../evaluate_service)， 拷贝到评估服务器的任意路径下。( 注：该路径下`sample`代码是在华为海思公开样例代码基础上进行开发的。)

- 根据实际环境修改配置文件`config.py`：
  - `davinci_environment_type` 为Davinci 环境类型， 当前支持`ATLAS200DK`, `ATLAS300` 以及开发板`Evb`环境， 请根据实际环境配置。
  - `ddk_host_ip`为评估服务器的ip地址， 请根据实际情况配置, `listen_port`为监听端口号， 可以任意设置， 注意不要与已有端口号冲突。
  - 如果环境类型是`ATLAS200DK`, 还需要配置如下字段， 如果环境类型不是`ATLAS200DK`， 可忽略如下配置。`ddk_user_name`为登录评估服务器的用户名， `atlas_host_ip`为`ATLAS200DK`硬件的实际IP地址。
- 运行`install.sh`脚本即可完成依赖环境的安装的评估服务的启动。

## 4. 注意事项

1. 如果您使用的是`Pytorch`框架， 在评估服务的客户端需要进行`Pytorch`模型的转换， 使用了第三方开源软件， 请自行获取并放在`./third_party`目录下。 开源软件下载地址： <https://github.com/xxradon/PytorchToCaffe>
2. 如果您使用的`Pytorch`版本在1.2及以下， 在`Pytorch`模型转换为`onnx`模型时可能会遇到算子不支持的情况。 如`upsample_bilinear2d`算子不支持， 您可以选择升级`Pytorch`版本到1.3及以上， 或者您可以从`Pytorch`官方代码库中获取`pytorch/torch/onnx/symbolic_opset10.py`, 拷贝到对应的`Pytorch`安装目录下。
