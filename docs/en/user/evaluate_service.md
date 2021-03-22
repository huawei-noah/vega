# Evaluate Service

## 1.  Introduction

The model evaluation service is used to evaluate the performance of a model on a specific hardware device, such as the accuracy, model size, and latency of a pruned and quantized model on the Atlas 200 DK.

Currently, the evaluation service supports Davincit inference chips (Atlas 200 DK, ATLAS300, and development board environment Evb) and mobile phones. More devices will be supported in the future.

The evaluation service uses the CS architecture. The evaluation service is deployed on the server. The client sends an evaluation request to the server through the `REST` interface and obtains the result. Vega can use the evaluation service to detect model performance in real time during network architecture search. After a candidate network is generated in the search phase, the network model can be sent to the evaluation service. After the model evaluation is complete, the evaluation service returns the evaluation result to Vega. Vega performs subsequent search based on the evaluation result. This real-time evaluation on the actual device helps to search for a network structure that is more friendly to the actual hardware.

## 2. spec

Supported Models and Hardware Devices:

| Algorithm | Model | Atlas 200 DK |Atlas 300 | Bolt |
| :--: | :--: | :--: | :--: | :--: |
| Prune-EA | ResNetGeneral | supported | supported | supported |
| ESR-EA | ESRN | | supported | supported |
| SR-EA | MtMSR | | supported | supported |
| Backbone-nas | ResNet| supported| supported | |
| CARS | CARSDartsNetwork | | supported | |
| Quant-EA | ResNetGeneral | supported | supported | supported |
| CycleSR | CycleSRModel | | | |
| Adlaide-EA | AdelaideFastNAS | | supported | |
| Auto-Lane | ResNetVariantDet | | |
| Auto-Lane | ResNeXtVariantDet | | |

## 3. Evaluation Service Deployment

### 3.1 Environment Installation and Configuration (Optional)

Configure the hardware (Atlas 200 DK, Atlas 300, or mobile phone) by following the instructions provided in the following sections.

### 3.1.1 Installing the Atlas 200 DK Environment (Optional)

#### 3.1.1.1 Preparations

1. An 8 GB or larger SD card and a card reader are available.
2. A server where Ubuntu 16.04.3 has been installed.
3. Download the system image: [ubuntu-16.04.3-server-arm64.iso](http://old-releases.ubuntu.com/releases/16.04.3/ubuntu-16.04.3-server-arm64.iso)
4. Download the make_sd_card.py and make_ubuntu_sd.sh from <https://github.com/Ascend-Huawei/tools>.
5. Download the developer running package mini_developerkit-1.3.T34.B891.rar from <https://www.huaweicloud.com/ascend/resources/Tools/1>.
6. Decompress the developer package and upload it to the user directory.

#### 3.1.1.2 Installing and Configuring the Atlas200 DK

1. Insert the SD card into the card reader and connect the card reader to the USB port on the Ubuntu server.
2. Install dependencies on the Ubuntu server:

    ```bash
    apt-get install qemu-user-static binfmt-support python3-yaml gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    ```

3. Run the following command to query the name of the USB device where the SD card is located:

    ```bash
    fdisk -l
    ```

4. Run the SD card making script to make a card. The USB device name is the name obtained in the previous step.

    ```bash
    python3 make_sd_card.py  local  USB Device Name
    ```

5. After the card is created, remove the SD card from the card reader, insert the SD card into the card slot of the Atlas 200 DK developer board, and power on the Atlas 200 DK developer board.

#### 3.1.1.3 Installing and Configuring the Evaluation Server Environment

1. Downloading and Installing the DDK Package and Synchronizing the Library

    Download address: <https://www.huaweicloud.com/ascend/resources/Tools/1>
    For details about the installation procedure, see the official document: <https://www.huaweicloud.com/ascend/doc/atlas200dk/1.32.0.0(beta)/zh/zh-cn_topic_0204690657.html>

2. Configuring the Cross Compilation Environment
    To install the compilation environment required by the Atlas 200 DK on the evaluation server, run the following command:

    ```bash
    sudo apt-get install g++-aarch64-linux-gnu
    ```

3. Configure the following environment variables in `/etc/profile` of the server. The value of `/home/<user name>` in the file must be a specific path.

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

4. Configuring SSH Mutual Trust
    File transfer and remote command execution are required between the evaluation server and the Atlas 200 DK. Therefore, you need to configure SSH mutual trust in the two environments to ensure that the script can be automatically executed.

    a.  Install the SSH. `sudo apt-get install ssh`
    b.  Generate a key. The `ssh-keygen -t rsa` command generates the id_rsa and id_rsa.pub files in the ~/.ssh/ directory. id_rsa.pub is the public key.
    c.  Check the authorized_keys file in the directory. If the file does not exist, create it and run the `chmod 600 ~/.ssh/authorized_keys` command to change the permission.
    d.  Copy the public key. Copy the content of the public key id_rsa.pub to the authorized_keys file on another host.
    **Note**: Perform the preceding steps on the evaluation server and Atlas 200 DK separately to ensure SSH trust between the two servers.

### 3.1.2 Installing and Configuring the Atlas 300 Environment (Optional)

For details, see the Huawei official tutorial at <https://support.huawei.com/enterprise/zh/ai-computing-platform/a300-3000-pid-250702915>.

Note: The preceding documents may be updated. Please follow the released updates or obtain the corresponding guide documents. After the environment is installed, you need to set environment variables. For details, see the preceding guide. To facilitate environment configuration, we provide the environment variable configuration template [env_atlas300.sh](https://github.com/huawei-noah/vega/blob/master/evaluate_service/hardwares/davinci/env/env_atlas300.sh) for your reference. The actual environment prevails.

The installation of the Atlas300 environment is complex. To ensure that the environment is correctly installed, please run [check_atlas300.sh](https://github.com/huawei-noah/vega/blob/master/evaluate_service/hardwares/davinci/env/check_atlas300.sh).

### 3.1.3 Installing and Configuring the Mobile Phone Environment (Optional)

#### 3.1.3.1 Preparations

1. Prepare a Kirin 980 mobile phone. Nova 5 is recommended.
2. A server where Ubuntu 16.04.3 has been installed.

#### 3.1.3.2 Installing and Configuring the Evaluation Server and Mobile Phone

1. Install the adb tool on the Linux server.

    ```bash
    apt install adb
    ```

2. Connect the mobile phone to the evaluation server through the USB port, enable the developer option, and run the following command on the evaluation server:

    ```bash
    adb devices
    ```

    If the following information is displayed, the connection is successful:

    ```text
    List of devices attached
    E5B0119506000260 device
    ```

#### 3.1.3.3 Handling Device Connection Failures

If you cannot obtain the device by running the `adb devices` command on the server, perform the following steps to connect to the device:

1. Run the `lsusb` command on the evaluation server. The device list is displayed. Find the device ID.

2. Edit the 51-android.rules file.

    ```bash
    sudo vim /etc/udev/rules.d/51-android.rules
    ```

    Write the following content:

    ```text
    SUBSYSTEM=="usb", ATTR{idVendor}=="12d1", ATTR{idProduct}=="107e", MODE="0666"
    ```

    Note: 12d1 and 107e are the IDs queried in the previous step.

3. Edit the adb_usb.ini file.

    ```bash
    vim -/.android/adb_usb.ini
    ```

    Write the following content:

    ```text
    0x12d1
    ```

    Note: 12d1 is the ID queried in step 5.1.

4. Restart the ADB service.

    ```bash
    sudo adb kill-server
    sudo adb start-server
    ```

5. Run the `adb devices` command again to check whether the connection is successful.


### 3.1.4 Installing and Configuring the NPU Environment for Kirin 990 Mobile Phones (Optional)
3.1.4.1 Preparations
1. Prepare a Kirin 990 phone. The Mate30 Pro is recommended.
2. A server on which ubuntu 16.04.3 has been installed.

3.1.4.2 Installation and Deployment  
1 Download the HUAWEI HiAI DDK from https://developer.huawei.com/consumer/cn/doc/development/hiai-Library/ddk-download-0000001053590180, Download hwhiai-ddk-100.500.010.010.zip, and decompress it to the /data/tools/ directory. The directory structure is "/data/tools/hwhiai-ddk-100.500.010.010/".    
2 Copy the dependent files to the mobile phone.  
Copy all contents in the tools_sysdbg directory to the /data/local/tmp directory on the mobile phone.
```bash
adb push /data/tools/hwhiai-ddk-100.500.010.010/tools/tools_sysdbg/* /data/local/tmp/
``` 
3 Log in to the mobile phone, set environment variables, and add the file execution permission.
```bash
adb shell
export LD_LIBRARY_PATH=/data/local/tmp/
chmod +x /data/local/tmp/model_run_tool
chmod +x /data/local/tmp/data_proc_tool
```
4 Installing the ADB Debug Tool  
Reference to section 3.1.3.2.

### 3.2 Installing and Starting the Evaluation Service

Download the code [evaluate_service](https://github.com/huawei-noah/vega/tree/master/evaluate_service) to any directory on the evaluation server. (Note: The `sample` code in this path is developed based on the Huawei HiSilicon public sample code.)

- Modify the `config.py` configuration file based on the site requirements.
  - davinci_environment_type indicates the Davinci environment type. Currently, the `ATLAS200DK`, `ATLAS300` and development board `Evb` are supported. Set this parameter based on the site requirements.
  - `ddk_host_ip` indicates the IP address of the evaluation server. Set it based on the site requirements. `listen_port` indicates the listening port number, which can be set to any value. Ensure that it does not conflict with an existing port number.
  - If the environment type is `ATLAS200DK`, you need to configure the following fields. If the environment type is not `ATLAS200DK`, ignore the following configuration. `ddk_user_name` is the user name for logging in to the evaluation server, and `atlas_host_ip` is the actual IP address of the `ATLAS200DK` hardware.
- Run the `install.sh` script to start the evaluation service that depends on the environment installation.

## 4. Use evaluate service

To use evaluate service, you only need to configure a few lines in the configuration file, as shown in the following example.

```yaml
evaluator:
    type: Evaluator
    device_evaluator:
        type: DeviceEvaluator
        hardware: "Davinci"
        remote_host: "http://192.168.0.2:8888"
```

The configuration of `evaluator` is at the same level as your configuration of `trainer`. Two parameters need to be configured. `hardware` indicates the hardware device to be evaluated. Currently, `Davinci` and `Bolt` are supported. `remote_host` indicates the IP address and port number of the evaluation server to be deployed.

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

## 6. FAQ

### 6.1 Pytorch Model Evaluation

The `Pytorch` model needs to be converted on the Appraisal client. Download [PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe) and store it in the `./third_party` directory (the third_party directory and vega directory are at the same directory level).

Note: The third-party open-source software does not support pytorch1.1. If you use the model in the native torchvisoin and the torchvision version is later than 0.2.0, you need to make the following additional modifications:
Add the following content to the `pytorch_to_caffe.py` file:

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

### 6.2 Model evaluation of Pytorch 1.2 and earlier versions

If the `Pytorch` version is 1.2 or earlier, operators may not be supported when the `Pytorch` model is converted to the `onnx` model.  If the `upsample_bilinear2d` operator is not supported, you can upgrade the `Pytorch` version to 1.3 or later, or you can obtain `pytorch/torch/onnx/symbolic_opset10.py`, from the `Pytorch` official code library and copy it to the `Pytorch` installation directory.

### 6.3 Failed to find scripts such as model_convert.sh

There are many `shell` scripts in the evaluation service. The file format must be `unix`. If you have opened a file in Windows or converted the file when downloading the code, the file format may be changed to DOS. Pay attention to the file format.
