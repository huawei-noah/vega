# FAQ

## 1. Exceptions

### 1.1 Exception `ModuleNotFoundError: No module named 'mmdet'`

To run algorithms such as SP-NAS, you need to install the open-source software mmdetection. For details, see the installation guide of the software.

### 1.2 Exception `ModuleNotFoundError: No module named 'nasbench'`

Before running the benchmark, install the open-source software NASBench. For details, see the installation guide of the software.

### 1.3 Exception `Exception: Failed to create model, model desc={<model desc>}`

The possible causes are as follows:

1. The network is not registered with the Vega. Before invoking the network, you need to use `@ClassFactory.register` to register the network. For details, see <https://github.com/huawei-noah/vega/tree/master/examples/fully_train/fmd>.
2. The model description file of the network is incorrect. You can locate the fault based on `<model desc>` in the exception information.

### 1.5 Exception `ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory`

The opencv-python system dependency library is missing. Run the following command:

```bash
sudo apt install libglib2.0-0
```

### 1.6 Exception `ModuleNotFoundError: No module named'skbuild '` or stuck in `Running setup.py bdist_wheel for opencv-python-headless...` during installation

The possible cause is that the PIP version is too early. Run the following command:

```bash
pip3 install --user --upgrade pip
```

### 1.7 Exception `PermissionError: [Errno 13] Permission denied: 'dask-scheduler'` or `FileNotFoundError: [Errno 2] No such file or directory: 'dask-scheduler': 'dask-scheduler'`

This type of exception is usually caused by the failure to find `dask-scheduler` in `PATH`. Generally, the file is installed in `/<user home path>/.local/bin`.
After the Vega is installed , `/<user home path>/.local/bin/` is automatically added to the `PATH` environment variable. The setting does not take effect immediately. You can run the ls command `source ~/.profile` or log in again to make the setting take effect.
If the problem persists, check whether the dask-scheduler file exists in the `/<user home path>/.local/bin` directory. 
If the file already exists, manually add `/<user home path>/.local/bin` to the environment variable `PATH`.

### 1.8 Exception During Pytorch model evaluation: `FileNotFoundError: [Errno 2] No such file or directory: '<path>/torch2caffe.prototxt'`

For details, see section 6.1 in [Evaluate Service](./evaluate_service.md).

## 2. Common Configuration Problems

### 2.1 How do I configure multi-GPU/NPU

If multiple GPUs or NPUs are deployed on the host running Vega, you can set the following configuration items to support multiple GPUs or NPUs:

```yaml
general:
    parallel_search: True
    parallel_fully_train: True
    devices_per_trainer: 1
```

Where:

- parallel_search：Controls whether multiple models are searched in parallel during the model search phase, each of which uses one or more GPUs/NPUs.
- parallel_fully_train: Controls whether to train multiple models concurrently in the Fully Train phase. Each model uses one or more GPUs or NPUs.
- devices_per_trainer: If any of the preceding parameters is set to True, this parameter specifies the number of GPUs/NPUs corresponding to a model.

Note: The CARS and DARTS algorithms do not support parallel search.

### 2.2 How do I specify the GPU environment where Vega runs

If there are multiple GPUs in the running environment, run the following command to control the GPUs used by Vega:

Using a single GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vega.pipeline ./nas/backbone_nas/backbone_nas.yml
```

Using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=2,3 python3 -m vega.pipeline ./nas/backbone_nas/backbone_nas.yml
```

### 2.3 How Do I Load a Pre-training Model by Modifying Configuration Items

You can load the pre-training model by modifying the configuration item. For example, load the pre-training model simple_cnn.pth.

```yaml
    model:
        model_desc:
            modules: [backbone]
            backbone:
                type: SimpleCnn
                num_class: 10
                fp16: False
        pretrained_model_file: "./simple_cnn.pth"
```

### 2.4 How do I view logs and set log levels

By default, Vega logs are stored in the following path:

```text
./tasks/<task id>/logs
```

To configure the log level, modify the following configuration items:

```yaml
general:
    logger:
        level: info  # debug|info|warn|error|
```

### 2.5 How do I view the search progress in real time

Vega provides the visualized progress of the model search process. User could set `VisualCallBack` within `USER.yml` as follow,

```yaml
    trainer:
        type: Trainer
        callbacks: [VisualCallBack, ]
```

The output directory of the visualized information is as follows:

```text
./tasks/<task id>/visual
```

Run the `tensorboard --logdir PATH` command on the active node to start the service and view the progress in the browser. For details, see TensorBoard commands and instructions.

### 2.6 如何终止后台运行的vega程序

Vega在多个GPU/NPU场景中，会启动dask scheduler、dask worker及训练器，若仅仅杀死Vega主进程会造成部分进程不会及时的关闭，其占用的资源一直不会被释放。

可使用如下命令终止Vega应用程序：

```bash
# 查询运行中的Vega主程序的进程ID
vega-kill -l
# 终止一个Vega主程序及相关进程
vega-kill -p <pid>
# 或者一次性的终止所有Vega相关进程
vega-kill -a
# 若主程序被非常正常关闭，还存在遗留的相关进程，可使用强制清理
vega-kill -f
```

### 2.6 How Do I Stop the Vega Program Running in the Background?

In the multi-GPU/NPU scenario, Vega starts the dask scheduler, dask worker, and trainer. If only the main Vega process is killed, some processes are not stopped in time and the resources occupied by these processes are not released.

Run the following command to stop the Vega application:

```bash
# Query the process ID of the running Vega main program.
vega-kill -l
# Stop a Vega main program and related processes.
vega-kill -p <pid>
# Or stop all Vega processes at a time.
vega-kill -a
# If the main program is closed normally and there are still residual processes, you can forcibly clear the process.
vega-kill -f
```
