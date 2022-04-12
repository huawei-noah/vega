# FAQ

## 1. Exceptions

### 1.1 Exception `Exception: Failed to create model, model desc={<model desc>}`

The possible causes are as follows:

1. The network is not registered with the Vega. Before invoking the network, you need to use `@ClassFactory.register` to register the network. For details, see <https://github.com/huawei-noah/vega/tree/master/examples/fully_train/fmd>.
2. The model description file of the network is incorrect. You can locate the fault based on `<model desc>` in the exception information.

### 1.2 Exception `ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory`

The opencv-python system dependency library is missing. Run the following command:

```bash
sudo apt install libglib2.0-0
```

### 1.3 Exception `ModuleNotFoundError: No module named'skbuild '` or stuck in `Running setup.py bdist_wheel for opencv-python-headless...` during installation

The possible cause is that the PIP version is too early. Run the following command:

```bash
pip3 install --user --upgrade pip
```

### 1.4 Exception `PermissionError: [Errno 13] Permission denied: 'dask-scheduler'`, `FileNotFoundError: [Errno 2] No such file or directory: 'dask-scheduler': 'dask-scheduler'`, or `vega: command not found`

This type of exception is usually caused by the failure to find `dask-scheduler` in `PATH`. Generally, the file is installed in `/<user home path>/.local/bin`.
After the Vega is installed , `/<user home path>/.local/bin/` is automatically added to the `PATH` environment variable. The setting does not take effect immediately. You can run the ls command `source ~/.profile` or log in again to make the setting take effect.
If the problem persists, check whether the dask-scheduler file exists in the `/<user home path>/.local/bin` directory. 
If the file already exists, manually add `/<user home path>/.local/bin` to the environment variable `PATH`.

## 2. Configuration Issues

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
- devices_per_trainer: If any of the preceding parameters is set to True, this parameter specifies the number of GPUs corresponding to a model.

Note: The CARS, DARTS, and ModularNAS algorithms do not support parallel search.

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

### 2.5 How Do I Stop the VEGA Program Running in the Background

If only the main Vega process is killed, some processes will not be stopped in time, and the resources occupied by the processes will not be released.

In safe mode, the Vega application can be terminated using the following command:

```bash
# Query the process ID of the running Vega main program.
vega-process -s
# Terminate a Vega main program and related processes.
vega-kill -s -p <pid>
# Terminate a Vega main program and related processes.
vega-kill -s -t <task id>
# Or stop all Vega-related processes at a time.
vega-kill -s -a
# If the main program is shut down normally and there are remaining related processes, you can forcibly clear the process.
vega-kill -s -f
```

In common mode, run the following command:：

```bash
vega-process
vega-kill -p <pid>
vega-kill -a
vega-kill -f
```

### 2.6 How Do I Query the Running Vega Program

In safe mode, run the following command to query the running Vega applications:

```bash
vega-process -s
```

In common mode, you can run the following command to query:

```bash
vega-process
```

### 2.7 How Do I Query the Vega Program Running Progress

In safe mode, you can run the following command to query the running progress of the Vega program:

```bash
vega-progress -s -t <Task ID> -r <Task Root Path>
```

In common mode, you can run the following command to query:

```bash
vega-progress -t <Task ID> -r <Task Root Path>
```

### 2.8 How to Perform Model Inference Using the Vega Program

Classification model inference can be performed with the command `vega-inference`, and detection model inference can be performed with the command `vega-inference-det`.

Run the following command to query the command parameters:

```bash
vega-inference --help
vega-inference-det --help
```

## 3. Precautions

### 3.1 Reserve Sufficient Disk Space

During Vega running, there is a model that caches each searched network. When the number of searched networks is large, a large amount of storage space is required. Reserve sufficient disk space based on the number of search network models for each search algorithm.
