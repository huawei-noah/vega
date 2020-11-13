# FAQ

## 1. Runtime exceptions

### 1.1 Exception `ModuleNotFoundError: No module named 'mmdet'`

To run algorithms such as SM-NAS and SP-NAS, you need to install the open-source software mmdetection. For details, see the installation guide of the software.

### 1.2 Exception `ModuleNotFoundError: No module named 'nasbench'`

Before running the benchmark, install the open-source software NASBench. For details, see the installation guide of the software.

### 1.3 Exception `ModuleNotFoundError: No module named '<module name>'`

After Vega is installed using the `pip3 install noah-vega --user` command, the third-party open-source software on which Vega depends is not installed. You need to run the `python3 -m vega.tools.install_pkgs` command to install the third-party open-source software. You can view the installation options through `python3 -m vega.tools.install_pkgs -h`. The `install_dependencies.sh` file is generated in the current directory to facilitate troubleshooting and commissioning during the installation.

### 1.4 Exception `Exception: Failed to create model, model desc={<model desc>}`

The possible causes are as follows:

1. The network is not registered with the Vega. Before invoking the network, you need to use `@ClassFactory.register` to register the network. For details, see <https://github.com/huawei-noah/vega/tree/master/examples/fully_train/fmd>.
2. The model description file of the network is incorrect. You can locate the fault based on `<model desc>` in the exception information.

### 1.5 Exception `ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory`

The opencv-python system dependency library is missing. Run the following command to rectify the fault:

```bash
sudo apt install libglib2.0-0
```

### 1.6 Exception `PermissionError: [Errno 13] Permission denied: 'dask-scheduler'`

This type of exception is usually caused by the failure to find `dask-scheduler` in `PATH`. Generally, the file is installed in `/<user home path>/.local/bin`.
After the Vega is installed and `python3 -m vega.tools.install_pkgs` is executed, `/<user home path>/.local/bin/` is automatically added to the `PATH` environment variable. The setting does not take effect immediately. The setting takes effect only after the user logs in to the server again.
If the problem persists after you log in to the server again, check whether the dask-scheduler file exists in the `/<user home path>/.local/bin` directory. If the file does not exist, run the `python3 -m vega.tools.install_pkgs` command again and ensure that no exception occurs during the installation.
If the file already exists, manually add `/<user home path>/.local/bin` to the environment variable `PATH`.

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
CUDA_VISIBLE_DEVICES=1 python3 ./run_pipeline.py ./nas/backbone_nas/backbone_nas.yml
```

Using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=2,3 python3 ./run_pipeline.py ./nas/backbone_nas/backbone_nas.yml
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

```
    trainer:
        type: Trainer
        callbacks: [VisualCallBack, ]
```



The output directory of the visualized information is as follows:

```text
./tasks/<task id>/visual
```

Run the `tensorboard --logdir PATH` command on the active node to start the service and view the progress in the browser. For details, see TensorBoard commands and instructions.
