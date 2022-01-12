# Vega 评估服务

**中文 | [English](./README.md)**

---

## 1. 简介

模型评估服务是用于评估模型在特定硬件设备上的性能，如评估剪枝和量化后的模型在Atlas200 DK、Atlas300上的准确率、模型大小和时延等。

评估服务目前支持的硬件设备为Davinci推理芯片（Atlas200 DK、ATLAS300产品和开发板环境Evb)和手机，后继会扩展支持更多的设备。

评估服务为CS架构， 评估服务在服务端部署， 客户端通过`REST`接口向服务端发送评估请求和获取结果。Vega在进行网络架构搜索时，可以利用评估服务进行实时检测模型性能。在搜索阶段产生备选网络后，可以将该网络模型发送给评估服务，评估服务完成模型评估后，返回评估结果给Vega，Vega根据评估结果，进行后继的搜索。这种实时的在实际的设备上的评估，有利于搜索出对实际硬件更加友好的网络结构。

## 2. 规格

支持的模型和硬件设备

| 算法 | 模型 | Atlas 200 DK |Atlas 300 | Bolt |
| :--: | :--: | :--: | :--: | :--: |
| Prune-EA | ResNetGeneral | √ | √ | √|
| ESR-EA | ESRN | | √ | √ |
| SR-EA | MtMSR | | √ | √ |
| Backbone-nas | ResNet | √ | √ | |
| CARS | CARSDartsNetwork | | √ | |
| Quant-EA | ResNetGeneral | √ | √ | √ |
| CycleSR | CycleSRModel | | | |
| Adlaide-EA | AdelaideFastNAS | | √ | |
| Auto-Lane | ResNetVariantDet | | |
| Auto-Lane | ResNeXtVariantDet | | |

## 3. 评估服务部署

以下介绍Atalas 300评估服务的部署过程，若需要部署Atlas 200DK或者ARM芯片手机，请联系我们。

### 3.1 安装配置Atlas300环境

首先需要配置Ascend 300环境，请参考[配置文档](./ascend_310.md)。

然后请安装评估服务，请执行如下命令安装：

```bash
pip3 install --user --upgrade evaluate-service
```

安装完成后，将`~/.local/lib/python3.7/site-packages/evaluate_service/hardwares/davinci/samples/atlas300`拷贝到当前目录，执行如下操作，检查环境是否配置正确：

```bash
echo "[INFO] start check the enviroment..."
python3 -c "import te" && echo "[INFO] check te sucess"
python3 -c "import topi" && echo "[INFO] check topi sucess"
atc --version && echo "[INFO] check atc sucess "
echo "[INFO] start compile the example..."
cd ./atlas300/
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make  && echo "[INFO] check the env sucess!"
```

### 3.2 启动评估服务

使用如下命令启动评估服务：

```shell
vega-evaluate_service-service -i {your_ip_adress} -p {port} -w {your_work_path}
```

其中：

- `-i`参数指定当前使用的服务器的ip地址
- `-p`参数指定当前使用的服务器的的监听端口，默认值8888
- `-w`参数指定工作路径， 程序运行时的中间文件将存储在该目录下，请使用绝对路径

注意：

以上启动命令会启动安全模式，需要预先进行安全配置，请参考[安全配置](https://github.com/huawei-noah/vega/tree/master/docs/cn/user/security_configure.md)。

也可以使用`-s`参数，启用普通模式，不需要如上配置，命令如下：

```shell
vega-evaluate_service-service -s -i {your_ip_adress} -w {your_work_path}
```

## 4. 使用评估服务

使用评估服务时， 需要在Vega调用的配置文件中做如下配置：

```yaml
evaluator:
    type: Evaluator
    device_evaluator:
        type: DeviceEvaluator
        hardware: "Davinci"
        remote_host: "https://<ip>:<port>"
```

其中：

- `evaluator`的配置和`trainer`配置处于同一层级。
- `hardware`为评估的硬件设备，当前支持`Davinci`和`Bolt`两种。
- `remote_host`为评估服务器的ip和端口号，对于普通模式，请设置为：`http://<ip>:<port>`

## 5. 自定义评估服务

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
