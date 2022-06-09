# Vega Evaluate Service

**English | [中文](./README.cn.md)**

---

## 1.  Introduction

The model evaluation service is used to evaluate the performance of a model on a specific hardware device, such as the accuracy, model size, and latency of a pruned and quantized model on the Atlas 200 DK, and Ascend 310 Series.

Currently, the evaluation service supports Davinci inference chips (Atlas 200 DK, Ascend 310 Series, and development board environment Evb) and mobile phones. More devices will be supported in the future.

The evaluation service uses the CS architecture. The evaluation service is deployed on the server. The client sends an evaluation request to the server through the `REST` interface and obtains the result. Vega can use the evaluation service to detect model performance in real time during network architecture search. After a candidate network is generated in the search phase, the network model can be sent to the evaluation service. After the model evaluation is complete, the evaluation service returns the evaluation result to Vega. Vega performs subsequent search based on the evaluation result. This real-time evaluation on the actual device helps to search for a network structure that is more friendly to the actual hardware.

## 2. spec

Supported Models and Hardware Devices:

| Algorithm | Model | Atlas 200 DK | Ascend 310 Series | Bolt |
| :--: | :--: | :--: | :--: | :--: |
| Prune-EA | ResNetGeneral | √ | √ | √ |
| ESR-EA | ESRN | | √ | √ |
| SR-EA | MtMSR | | √ | √ |
| Backbone-nas | ResNet| √| √ | |
| CARS | CARSDartsNetwork | | √ | |
| Quant-EA | ResNetGeneral | √ | √ | √ |
| CycleSR | CycleSRModel | | | |
| Adlaide-EA | AdelaideFastNAS | | √ | |
| Auto-Lane | ResNetVariantDet | | |
| Auto-Lane | ResNeXtVariantDet | | |

## 3. Evaluation Service Deployment

### 3.1 Environment installation and configuration (Optional)

Configure the hardware (Atlas 200 DK, Ascend 310 Series, or mobile phone) by following the instructions provided in the following sections.

### 3.1.1 Install the Atlas 200DK environment (Optional)

Please contact us.

### 3.1.2 Install and configure the Ascend 310 Environment (Optional)

Please refer to [configuration documentation](./docs/en/ascend_310.md).
For details, see the Huawei official tutorial at <https://support.huawei.com/enterprise/zh/ai-computing-platform/a300-3000-pid-250702915>.

Note: The preceding documents may be updated. Please follow the released updates or obtain the corresponding guide documents. After the environment is installed, you need to set environment variables. For details, see the preceding guide. To facilitate environment configuration, we provide the environment variable configuration template [env_atlas300.sh](https://github.com/huawei-noah/vega/blob/master/evaluate_service/hardwares/davinci/env/env_atlas300.sh) for your reference. The actual environment prevails.

The installation of the Ascend 310 environment is complex. To ensure that the environment is correctly installed, please run [check_atlas300.sh](https://github.com/huawei-noah/vega/blob/master/evaluate_service/hardwares/davinci/env/check_atlas300.sh).

### 3.1.3 Install and configure the mobile environment (Optional)

Please contact us.

### 3.1.4 Install and configure the NPU environment for Kirin 990 mobile (Optional)

Please contact us.

### 3.2 Configuration Inference Tool

You can download the code from [https://gitee.com/ascend/tools/tree/master/msame](https://gitee.com/ascend/tools/tree/master/msame), and compile it as an inference tool. The file name is `msame`.
Copy the compiled inference tool to the `~/.local/lib/python3.7/site-packages/evaluate_service/hardwares/davinci/` directory.

Note: The preceding inference tool comes from the Ascend community. We are not responsible for the security of this tool. You can determine whether to use this tool or configure other inference tool.

### 3.3 Start the evaluation service

Run the following command to start the evaluate service:

```shell
vega-evaluate_service-service -i {your_ip_adress} -p {port} -w {your_work_path} -t {davinci_environment_type} -s
```

where:

- `-i` indicates the IP of the server
- `-p` indicates the listen port,default is 8888
- `-w` indicates the work dir, please use the absolute path
- `-t` indicates the chipset model, default is Ascend310
- `-s` indicates the security mode

Note:
The above command will run in security mode, the security configurations need to be performed in advance.
please refer to [security cinfigure](https://github.com/huawei-noah/vega/tree/master/docs/cn/user/security_configure.md)。

You can also not use the `-s` parameter to enable the common mode. The security configuration is not required. The command is as follows:

```shell
vega-evaluate_service-service  -i {your_ip_adress} -w {your_work_path} -t {davinci_environment_type}
```

## 4. Use evaluate service

To use evaluate service, you only need to configure a few lines in the configuration file, as shown in the following example.

```yaml
evaluator:
    type: Evaluator
    device_evaluator:
        type: DeviceEvaluator
        hardware: "Davinci"
        remote_host: "https://<ip>:<port>"
```

where:

- `evaluator` is at the same level as your configuration of `trainer`. 
- `hardware` indicates the hardware device to be evaluated. Currently, `Davinci` and `Bolt` are supported. 
- `remote_host` indicates the IP address and port of the evaluation server. For common mode, please set as 
`http://<ip>:<port>`

## 5. Customizing the Evaluation Service (Optional)

Evaluate service supports devices such as Davinci inference chips and mobile phones. However, new hardware devices are emerging. Therefore, Vega provides customized scalability.

The process of the evaluate service is as follows:

1. obtaining input information
2. Instantiate a specific hardware instance according to the hardware to be evaluated
3. Model conversion
4. inference
5. Return the inference result

Steps 3 and 4 may be different for different hardware. Therefore, when new hardware needs to be added, perform the two steps based on the hardware usage. Specifically, the procedure is as follows:

Add a hardware class to the hardwares directory and implement the `convert_model` and `inference` interfaces as follows:

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

In the preceding example, the `MyHardware` class is defined and registered through `@ClassFactory.register()`.

The class implements the `convert_model` and `inference` interfaces, `backend` indicates the training framework through which the model is saved, for example, `pytorch` and `tensorflow`, which provide necessary auxiliary information for model parsing. `model` and `weight` indicate the training framework through which the model is saved, respectively.

Model and weight to be converted. The value of weight is optional and may be empty. `converted_model` and `input_data` indicate the converted model and input data, respectively.

Add the class to `__init__.py` of the hardware.

```python
from .my_hardware import MyHardware
```
