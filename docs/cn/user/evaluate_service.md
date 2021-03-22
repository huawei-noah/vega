# Evaluate Service

## 1. 简介

模型评估服务是用于评估模型在特定硬件设备上的性能，如评估剪枝和量化后的模型在Atlas 200 DK上的准确率、模型大小和时延等。

评估服务目前支持的硬件设备为Davinci推理芯片（Atlas200 DK、ATLAS300产品和开发板环境Evb)和手机，后继会扩展支持更多的设备。

评估服务为CS架构， 评估服务在服务端部署， 客户端通过`REST`接口向服务端发送评估请求和获取结果。Vega在进行网络架构搜索时，可以利用评估服务进行实时检测模型性能。在搜索阶段产生备选网络后，可以将该网络模型发送给评估服务，评估服务完成模型评估后，返回评估结果给Vega，Vega根据评估结果，进行后继的搜索。这种实时的在实际的设备上的评估，有利于搜索出对实际硬件更加友好的网络结构。

## 2. 规格

支持的模型和硬件设备

| 算法 | 模型 | Atlas 200 DK |Atlas 300 | Bolt |
| :--: | :--: | :--: | :--: | :--: |
| Prune-EA | ResNetGeneral | 支持 | 支持 | 支持|
| ESR-EA | ESRN | | 支持| 支持 |
| SR-EA | MtMSR | | 支持 | 支持|
| Backbone-nas | ResNet | 支持| 支持| |
| CARS | CARSDartsNetwork | | 支持 | |
| Quant-EA | ResNetGeneral | 支持 | 支持 | 支持|
| CycleSR | CycleSRModel | | | |
| Adlaide-EA | AdelaideFastNAS | | 支持 | |
| Auto-Lane | ResNetVariantDet | | |
| Auto-Lane | ResNeXtVariantDet | | |

## 3. 评估服务部署

### 3.1 环境安装配置（可选）

根据评估硬件（Atlas200 DK 、Atlas300、或者手机），分别按照如下章节指导配置。

### 3.1.1 安装 Atlas200 DK 环境（可选）

#### 3.1.1.1 准备工作

1. 准备好一张8GB以上SD卡及读卡器。
2. 已安装 ubuntu 16.04.3 的服务器一台。
3. 下载系统镜像： [ubuntu-16.04.3-server-arm64.iso](http://old-releases.ubuntu.com/releases/16.04.3/ubuntu-16.04.3-server-arm64.iso)
4. 下载制卡脚本： make_sd_card.py 和 make_ubuntu_sd.sh，下载地址：<https://github.com/Ascend-Huawei/tools>
5. 下载开发者运行包： mini_developerkit-1.3.T34.B891.rar，下载地址：<https://www.huaweicloud.com/ascend/resources/Tools/1>
6. 解压开发者运行包，并上传到用户目录下。

#### 3.1.1.2 安装和配置Atlas200 DK

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

3. 在服务器的 `/etc/profile` 中配置如下环境变量，注意文件中的`/home/<user name>`要配置为正确的路径：

    ```bash
    export DDK_PATH=/home/<user name>/huawei/ddk
    export PYTHONPATH=$DDK_PATH/site-packages/te-0.4.0.egg:$DDK_PATH/site-packages/topi-0.4.0.egg
    export LD_LIBRARY_PATH=$DDK_PATH/uihost/lib:$DDK_PATH/lib/x86_64-linux-gcc5.4
    export PATH=$PATH:$DDK_PATH/toolchains/ccec-linux/bin:$DDK_PATH/uihost/bin
    export TVM_AICPU_LIBRARY_PATH=$DDK_PATH/uihost/lib/:$DDK_PATH/uihost/toolchains/ccec-linux/aicpu_lib
    export TVM_AICPU_INCLUDE_PATH=$DDK_PATH/include/inc/tensor_engine
    export TVM_AICPU_OS_SYSROOT=/home/<user name>/tools/sysroot/aarch64_Ubuntu16.04.3
    export NPU_HOST_LIB=/home/<user name>/tools/1.32.0.B080/RC/host-aarch64_Ubuntu16.04.3/lib
    export NPU_DEV_LIB=/home/<user name>/tools/1.32.0.B080/RC/host-aarch64_Ubuntu16.04.3/lib
    ```

4. 配置ssh互信
    由于评估服务器和Atlas200 DK 之间需要进行文件传输以及远端命令的执行，因此需要分别在两个环境上配置ssh互信，确保脚本能够自动化运行。

    a. 安装ssh：`sudo apt-get install ssh`  
    b. 生成密钥：`ssh-keygen -t rsa` 会在~/.ssh/文件下生成id_rsa, id_rsa.pub两个文件，其中id_rsa.pub是公钥  
    c. 确认目录下的authorized_keys文件。若不存在需要创建， 并`chmod 600 ~/.ssh/authorized_keys`改变权限。  
    d. 拷贝公钥：分别将公钥id_rsa.pub内容拷贝到其他机器的authorized_keys文件中。  
    **注意**： 以上步骤需要在评估服务器和Atlas 200 DK 分别执行一遍， 确保这两台机器之间ssh互信。

### 3.1.2 安装配置Atlas300环境（可选）

参考华为图灵官方教程自行安装配置： <https://support.huawei.com/enterprise/zh/ai-computing-platform/a300-3000-pid-250702915> Atlas 300I 推理卡 用户指南（型号 3000）

注意：上述文档可能发生更新， 请及时关注我们发布的更新或自行获取得到相应的指导文档。环境安装后一般需要设置相应的环境变量， 请参考上述指导文档进行相应配置。为了方便您更好地进行环境配置， 我们提供了相关环境变量配置的模板 [env_atlas300.sh](https://github.com/huawei-noah/vega/blob/master/evaluate_service/hardwares/davinci/env/env_atlas300.sh) 供您参考， 请您以实际安装环境为准。

由于Atlas300环境安装较为复杂， 为了确保您的环境安装正确， 请您完成安装后运行检查环境脚本[check_atlas300.sh](https://github.com/huawei-noah/vega/blob/master/evaluate_service/hardwares/davinci/env/check_atlas300.sh)。

### 3.1.3 安装和配置手机环境(可选)

#### 3.1.3.1 准备工作

1. 准备Kirin 980手机1台，推荐Nova 5。
2. 已安装 ubuntu 16.04.3 的服务器一台。

#### 3.1.3.2 安装和配置评估服务器和手机

1. 在linux 系统服务器上安装adb工具。

    ```bash
    apt install adb
    ```

2. 通过USB端口将手机接入到评估服务器，并打开开发者选项，并在评估服务器上执行如下命令：

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

### 3.1.4 安装和配置麒麟990手机NPU环境（可选）
3.1.4.1 准备工作
1. 准备Kirin 990手机1台，推荐mate30 pro。
2. 已安装 ubuntu 16.04.3 的服务器一台。

3.1.4.2 安装和部署  
1 下载HUAWEI HiAI DDK, 下载链接：https://developer.huawei.com/consumer/cn/doc/development/hiai-Library/ddk-download-0000001053590180, 
选择下载hwhiai-ddk-100.500.010.010.zip, 下载后解压到"/data/tools/"目录下, 解压后目录结构为"/data/tools/hwhiai-ddk-100.500.010.010/"。  
2 拷贝相关依赖文件到手机  
把tools_sysdbg目录下所有内容拷贝到手机上的/data/local/tmp目录下  
```bash 
adb push /data/tools/hwhiai-ddk-100.500.010.010/tools/tools_sysdbg/*   /data/local/tmp/
```
3 进入到手机上， 设置环境变量, 添加文件执行权限
```bash
adb shell 
export LD_LIBRARY_PATH=/data/local/tmp/
chmod +x /data/local/tmp/model_run_tool
chmod +x /data/local/tmp/data_proc_tool
```
4 安装adb调试工具  
参考3.1.3.2节。

### 3.2 安装和启动评估服务

下载评估服务的代码[evaluate_service](https://github.com/huawei-noah/vega/tree/master/evaluate_service)， 拷贝到评估服务器的任意路径下。( 注：该路径下`sample`代码是在华为海思公开样例代码基础上进行开发的。)

- 根据实际环境修改配置文件`config.py`：
  - `davinci_environment_type` 为Davinci 环境类型， 当前支持`ATLAS200DK`, `ATLAS300` 以及开发板`Evb`环境， 请根据实际环境配置。
  - `ddk_host_ip`为评估服务器的ip地址， 请根据实际情况配置, `listen_port`为监听端口号， 可以任意设置， 注意不要与已有端口号冲突。
  - 如果环境类型是`ATLAS200DK`, 还需要配置如下字段， 如果环境类型不是`ATLAS200DK`， 可忽略如下配置。`ddk_user_name`为登录评估服务器的用户名， `atlas_host_ip`为`ATLAS200DK`硬件的实际IP地址。
- 运行`install_dependencies.sh`脚本可完成`Python`依赖环境的安装， 运行`run.sh`启动评估服务。

## 4. 使用评估服务

使用评估服务时， 只需要在配置文件中进行简单的几行配置即可， 如下面示例所示：

```yaml
evaluator:
    type: Evaluator
    device_evaluator:
        type: DeviceEvaluator
        hardware: "Davinci"
        remote_host: "http://192.168.0.2:8888"
```

`evaluator`的配置与您的`trainer`配置处于同一层级。其中需要配置的参数有2个， `hardware`为您指定的需要评估的硬件设备，当前支持`Davinci`和`Bolt`两种，
`remote_host`为您部署的评估服务器的ip和端口号。

## 5. 自定义评估服务（可选）

vega评估服务当前已经支持Davinci推理芯片和手机等端侧设备的评估， 但新的硬件设备是层出不穷的， 因此评估服务提供了可自定义的扩展能力。

评估服务的流程是：

1. 获取输入信息
2. 根据需要评估的硬件实例化一个具体的硬件实例
3. 模型转换
4. 推理
5. 返回推理结果

对于不同的硬件， 步骤3和4可能是不同的。 因此当需要添加新的硬件时， 需要根据具体硬件的用法实现这2个步骤。具体来说， 分以下几个步骤：

在hardwares目录下添加一个硬件类， 并实现`convert_model`和`inference`两个接口 如下：

```python
from class_factory import ClassFactory
@ClassFactory.register()
class MyHardware(object):

    def __init__(self, optional_params):
        pass

    def convert_model(self, backend, model, weight, **kwargs):
        pass

    def inference(self, converted_model, input_data, **kwargs):

        return latency, output
```

上面的示例中定义了`MyHardware`类， 并通过`@ClassFactory.register()`进行注册。 类中实现了`convert_model`和`inference`两个接口, `backend`表示模型是通过何种训练框架保存的， 如`pytorch`, `tensorflow`等， 为模型解析提供必要的辅助信息，`model`和`weight`分别表示需要转换的模型和权重，`weight`是非必须的，其值可能为空。`converted_model`和`input_data`分别表示转换之后的模型和输入数据。

然后在hardware的`__init__.py`中加入自定义的类。

```python
from .my_hardware import MyHardware
```

## 6. FAQ

### 6.1 Pytorch模型评估

在评估服务的客户端需要进行`Pytorch`模型的转换，请下载[PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe)获取并放在`./third_party`目录下(third_party目录与vega处于同一目录层级)。

注意： 该第三方开源软件不支持pytorch1.1版本， 并且如果您使用原生torchvisoin中的模型， 当torchvision版本高于0.2.0时， 您需要做以下额外修改:
修改`pytorch_to_caffe.py`文件， 增加以下内容：

```python

def _flatten(raw , input, * args):
    x = raw(input, *args)
    if not NET_INITTED:
        return x
    layer_name=log.add_layer(name='flatten')
    top_blobs=log.add_blobs([x],name='flatten_blob')
    layer=caffe_net.Layer_param(name=layer_name,type='Reshape',
                                bottom=[log.blobs(input)],top=top_blobs)
    start_dim = args[0]
    end_dim = len(x.shape)
    if len(args) > 1:
        end_dim = args[1]
    dims = []
    for i in range(start_dim):
        dims.append(x.shape[i])
    cum = 1
    for i in range(start_dim, end_dim):
        cum = cum * x.shape[i]
    dims.append(cum)
    if end_dim != len(x.shape):
        cum = 1
        for i in range(end_dim, len(x.shape)):
            cum = cum * x.shape[i]
        dims.append(cum)
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    log.cnet.add_layer(layer)
    return x


torch.flatten = Rp(torch.flatten,_flatten)
```

### 6.2 Pytorch 1.2版本及以下模型评估

如果您使用的`Pytorch`版本在1.2及以下， 在`Pytorch`模型转换为`onnx`模型时可能会遇到算子不支持的情况。 如`upsample_bilinear2d`算子不支持， 您可以选择升级`Pytorch`版本到1.3及以上， 或者您可以从`Pytorch`官方代码库中获取`pytorch/torch/onnx/symbolic_opset10.py`, 拷贝到对应的`Pytorch`安装目录下。

### 6.3 找不到`model_convert.sh`等脚本错误

评估服务中有很多`shell`脚本， 其文件格式应该为`unix`格式， 如果在windows上打开过相应文件， 或是`git`下载代码时进行了相应转换， 文件格式可能会变成`dos`格式， 需要转换为`unix`格式。
