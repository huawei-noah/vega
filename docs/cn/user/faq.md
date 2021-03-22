# FAQ

## 1. 常见异常汇总

### 1.1 异常 `ModuleNotFoundError: No module named 'mmdet'`

运行SP-NAS等算法时，需要单独安装开源软件mmdetection，具体安装步骤请参考该软件的安装指导。

### 1.2 异常 `ModuleNotFoundError: No module named 'nasbench'`

运行Benchmark时，需要单独安装开源软件NASBench，具体安装步骤请参考该软件的安装指导。

### 1.3 异常 `Exception: Failed to create model, model desc={<model desc>}`

出现该类问题的原因有两类：

1. 该网络未注册到Vega中，在调用该网络前，需要使用`@ClassFactory.register`注册该网络，可参考示例<https://github.com/huawei-noah/vega/tree/master/examples/fully_train/fmd>。
2. 该网络的模型描述文件有错误，可通过异常信息中的`<model desc>`定位问题的原因。

### 1.5 异常 `ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory`

该异常可能是因为opencv-python缺少了系统依赖库，可尝试使用如下命令解决：

```bash
sudo apt install libglib2.0-0
```

### 1.6 安装过程中出现异常 `ModuleNotFoundError: No module named 'skbuild'`，或者在安装过程中卡在`Running setup.py bdist_wheel for opencv-python-headless ...`

该异常可能是pip的版本过低，可尝试使用如下命令解决：

```bash
pip3 install --user --upgrade pip
```

### 1.7 异常 `PermissionError: [Errno 13] Permission denied: 'dask-scheduler'`, 或者 `FileNotFoundError: [Errno 2] No such file or directory: 'dask-scheduler': 'dask-scheduler'`

这类异常一般是因为在 `PATH` 路径中未找到 `dask-scheduler` ，一般该文件会安装在 `/<user home path>/.local/bin` 路径下。
在安装完 Vega ，会自动添加 `/<user home path>/.local/bin/` 到 `PATH` 环境变量中，但不会即时生效，需要该用户执行`source ~/.profile`，或者再次登录服务器后才会生效。
若问题还未解决，可先检查在 `/<user home path>/.local/bin` 路径下是否存在 `dask-scheduler` 文件。
若该文件已存在，则需要手动添加 `/<user home path>/.local/bin` 到环境变量 `PATH` 中。

### 1.8 Pytorch模型评估时，出现异常 `FileNotFoundError: [Errno 2] No such file or directory: '<path>/torch2caffe.prototxt'`

请参考文档 [Evaluate Service](./evaluate_service.md) 6.1 章节。

## 2. 常见配置问题汇总

### 2.1 如何配置多GPU/NPU支持

若运行Vega的主机环境中有多张GPU/NPU，可通过设置如下配置项支持多GPU/NPU：

```yaml
general:
    parallel_search: True
    parallel_fully_train: True
    devices_per_trainer: 1
```

其中：

- parallel_search：控制是否在模型搜索阶段并行搜索多个模型，其中每个模型使用一个或多个GPU/NPU。
- parallel_fully_train: 控制是否在fully train阶段并行训练多个模型，其中每个模型使用一个或多个GPU/NPU。
- devices_per_trainer: 当如上任一控制项为True是生效，用于控制一个模型对应多少个GPU/NPU。

注意：CARS和DARTS算法不支持并行搜索。

### 2.2 如何指定Vega运行的GPU环境

若运行环境中有多张GPU卡，可使用如下命令控制Vega使用哪些GPU：

使用单个GPU：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vega.pipeline ./nas/backbone_nas/backbone_nas.yml
```

使用多个GPU：

```bash
CUDA_VISIBLE_DEVICES=2,3 python3 -m vega.pipeline ./nas/backbone_nas/backbone_nas.yml
```

### 2.3 如何通过修改配置项加载预训练模型

可通过修改配置项加载预训练模型，如下例加载预训练模型simple_cnn.pth：

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

### 2.4 如何查看日志，如何设置日志级别

Vega的日志缺省保存在如下位置：

```text
./tasks/<task id>/logs
```

若要配置日志级别，可修改：

```yaml
general:
    logger:
        level: info  # debug|info|warn|error|
```

### 2.5 如何实时查看搜索进展

Vega提供了模型搜索过程可视化进展，用户只需在`USER.yml` 中配置`VisualCallBack`， 如下所示

```yaml
    trainer:
        type: Trainer
        callbacks: [VisualCallBack, ]
```

可视化信息输出目录为：

```text
./tasks/<task id>/visual
```

在主机上执行`tensorboard --logdir PATH`如下启动服务，在浏览器上查看进展。具体可参考tensorboard的相关命令和指导。

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
