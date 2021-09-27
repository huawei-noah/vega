# Configuration Reference

Vega decomposes the entire AutoML process from data to models into multiple steps, including network architecture search, hyperparameter optimization, data augmentation, and model training. Vega can combine these steps into a complete pipeline through configuration files and execute these steps in sequence, complete the entire process from data to model.

In addition, Vega designs a network and hyperparameter search space independent of the search algorithm for algorithms such as network architecture search, hyperparameter optimization, and data augmentation. You can adjust the configuration file to implement personalized search.

The following is an example of running the CARS algorithm:

```python
cd examples
vega ./nas/cars/cars.yml
```

The following describes each item in the configuration file.

## 1. Overall structure

The configuration of the vega can be divided into two parts:

1. General configuration. The configuration item name is `general`. It is used to set common and common configuration items, such as Backend, output path, and log level.
2. Pipeline configuration, including the following two parts:
   1. Pipeline definition. The configuration item name is pipeline, which is a list that contains all steps in the pipeline.
   2. Defines each step in Pipeline. The configuration item name is the name of each step defined in Pipeline.

```yaml
general:
    # general configuration

# Defining a Pipeline.
pipeline: [my_nas, my_hpo, my_data_augmentation, my_fully_train]

# defines each step. Refer to the following sections for details about
my_nas:
    # NAS configuration

my_hpo:
    # HPO configuration

my_data_augmentation:
    # Data augmentation configuration

my_fully_train:
    # fully train configuration
```

The following describes each configuration item in detail.

## 2. Public configuration items

The following public configuration items can be configured:

| Configuration item | Optional | Default value | Description |
| :--: | :-- | :-- | :-- |
| backend | pytorch \| tensorflow \| mindspore | pytorch | Backend.  |
| local_base_path | - | ./tasks/ | Working path. Each time when the system is running, a subfolder with time information (task id) is generated in the path. In this way, the output of multiple running is not overwritten. The task id subfolder contains two subfolders: output and worker. The output folder stores the output data of each step in the pipeline, and the worker folder stores temporary information.  <br> **In the clustered scenario, this path needs to be set to an EFS path that can be accessed by each computing node, and is used by different nodes to share data.** |
| timeout | - | 10 | Worker timeout interval, in hours. If the task is not completed within the interval, the worker is forcibly terminated. |
| parallel_search | True \| False | False | Whether to search multiple models in parallel. |
| parallel_fully_train | True \| False | False | Whether to train multiple models in parallel. |
| devices_per_trainer | 1..N (Tthe maximum number of GPUs or NPUs on a single node) | 1 | In parallel search and training, the number of devices (GPU \| NPU) allocated by each trainer, when parallel_search or parallel_fully_train is true. The default is 1, and each trainer is assigned one (gpu \| npu). |
| logger / level | debug \| info \| warn \| error \| critical | info | Log level |
| cluster / master_ip | - | ~ | In the cluster scenario, this parameter needs to be set to the IP address of the master node. |
| cluster / slaves | - | [] | In the cluster scenario, this parameter needs to be set to the IP address of other nodes except the master node. |
| quota | - | ~ | Models filter. Set maximum value or range of the floating-point calculation amount of the sampling model (MB), the parameters of the sampling model (KB), the latency of the sampling model (ms), max pipeline estimated running time set by user (hour). The options are "<", ">", "in", and "and".<br>eg: "flops < 10 and params in [100, 1000]" |

```yaml
general:
    backend: pytorch
    parallel_search: False
    parallel_fully_train: False
    devices_per_trainer: 1
    task:
        local_base_path: "./tasks"
    logger:
        level: info
    cluster:
        master_ip: ~
        slaves: []
    quota: "flops < 10 and params in [100, 1000]"
```

## 2.1 Parallel and distributed

During NAS/HPO search, one trainer corresponds to one GPU/NPU. If one trainer corresponds to multiple GPUs/NPUs, you can modify the `general.device_per_trainer` parameter.

Currently, this configuration works on PyTorch/GPU, as shown in the following:

```yaml
general:
    parallel_search: True
    parallel_fully_train: False
    devices_per_trainer: 2

pipeline: [nas, fully_train]

nas:
    pipe_step:
        type: SearchPipeStep
    search_algorithm:
        type: BackboneNas
        codec: BackboneNasCodec
    search_space:
        hyperparameters:
            -   key: network.backbone.depth
                type: CATEGORY
                range: [18, 34, 50]
            -   key: network.backbone.base_channel
                type: CATEGORY
                range:  [32, 48, 56]
            -   key: network.backbone.doublechannel
                type: CATEGORY
                range: [3, 4]
            -   key: network.backbone.downsample
                type: CATEGORY
                range: [3, 4]
    model:
        model_desc:
            modules: ['backbone']
            backbone:
                type: ResNet
                num_class: 10
    trainer:
        type: Trainer
    dataset:
        type: Cifar10

fully_train:
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/nas/"
    trainer:
        epochs: 160
        distributed: True
    dataset:
        type: Cifar10
```

In the fully training phase, Horovod (GPU) or HCCL (NPU) can be used to provide distributed data model training.

This is as follows:

```yaml
pipeline: [fully_train]

fully_train:
    pipe_step:
        type: HorovodTrainStep  # HorovodTrainStep(GPU), HcclTrainStep(NPU)
    trainer:
        epochs: 160
    model:
        model_desc:
            modules: ['backbone']
            backbone:
                type: ResNet
                num_class: 10
    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
```

## 3. NAS and HPO configuration items

HPO and NAS configuration items include:

| Configuration Item | Description |
| :--: | :-- |
| pipe_step / type | Set this parameter to `SearchPipeStep`, indicating that this step is a search step. |
| search_algorithm | Search algorithm configuration. For details, see the search algorithm section in this document. |
| search_space | Search space configuration. For details, see section "Search Space Configuration." |
| model | Model configuration. For details, see the search space section in this document. |
| dataset | Dataset configuration. For details, see the dataset section in this document. |
| trainer | Model training parameter configuration. For details, see the trainer section in this document. |
| evaluator | evaluator parameter configuration. For details, see the evaluator section in this document. |

The configuration:

```yaml
my_nas:
    pipe_step:
        type: SearchPipeStep
    search_algorithm:
        <search algorithm parameters>
    search_space:
        <search space parameters>
    model:
        <model parameters>
    dataset:
        <dataset parameters>
    trainer:
        <trainer parameters>
    evaluator:
        <evaluator parameters>
```

The following describes the search_algorithm and search_space configuration items.

### 3.1 Search Algorithm

Common search algorithms include the following configuration items:

| Configuration item | Description | Example |
| :--: | :-- | :-- |
| type | Search algorithm name. For details, see the configuration item in the example file of each algorithm. | `type: BackboneNas` |
| codec | Search algorithm encoder. Generally, an encoder is used with a search algorithm. | `codec: BackboneNasCodec` |
| policy | Search policy, which is a search algorithm parameter. | For example, if the BackboneNas uses the evolution algorithm, the policy is set to <br> `num_mutate: 10` <br> `random_ratio: 0.2` |
| range | Search range | For example, the search range of BackboneNas can be <br> `min_sample: 10` <br> `max_sample: 300` |

The search algorithm examples in the preceding table are as follows in the configuration file:

```yaml
search_algorithm:
    type: BackboneNas
    codec: BackboneNasCodec
    policy:
        num_mutate: 10
        random_ratio: 0.2
    range:
        max_sample: 300
        min_sample: 10
```

The search algorithm BackboneNas is used as an example. Configuration items vary according to search algorithms. For details, see the related chapters in the document of each search algorithm.

<table>
  <tr><th>Task</th><th>categorize</th><th>Algorithms</th></tr>
  <tr><td rowspan="3">Image Classification</td><td>Network Architecture Search</td><td><a href="../algorithms/cars.md">CARS</a>, <a href="../algorithms/nago.md">NAGO</a>, BackboneNas, DartsCNN, GDAS, EfficientNet</td></tr>
  <tr><td> Hyperparameter Optimization</td><td><a href="../algorithms/hpo.md">ASHA, BOHB, BOSS, PBT, Random</a></td></tr>
  <tr><td>Data Augmentation</td><td><a href="../algorithms/pba.md">PBA</a></td></tr>
  <tr><td rowspan="2">Model Compression</td><td>Model Pruning</td><td><a href="../algorithms/prune_ea.md">Prune-EA</a></td></tr>
  <tr><td>Model Quantization</td><td><a href="../algorithms/quant_ea.md">Quant-EA</a></td></tr>
  <tr><td rowspan="2">Image Super-Resolution</td><td>Network Architecture Search</td><td><a href="../algorithms/sr_ea.md">SR-EA</a>, <a href="../algorithms/esr_ea.md">ESR-EA</a></td></tr>
  <tr><td>Data Augmentation</td><td><a href="../algorithms/cyclesr.md">CycleSR</a></td></tr>
  <tr><td>Image Segmentation</td><td>Network Architecture Search</td><td><a href="../algorithms/adelaide_ea.md">Adelaide-EA</a></td></tr>
  <tr><td>Object Detection</td><td>Network Architecture Search</td><td><a href="../algorithms/sp_nas.md">SP-NAS</a></td></tr>
  <tr><td>Lane Detection</td><td>Network Architecture Search</td><td><a href="../algorithms/auto_lane.md">Auto-Lane</a></td></tr>
  <tr><td rowspan="2">Recommender System</td><td>Feature Selection</td><td><a href="../algorithms/autofis.md">AutoFIS</a></td></tr>
  <tr><td>Feature Interactions Selection</td><td><a href="../algorithms/autogroup.md">AutoGroup</a></td></tr>
</table>

### 3.1.1 HPO Search Algorithm Settings

Common configuration items for search algorithms such as Random, ASHA, BOHB, BOSS, and PBT are as follows:

|Configuration Item|Description|Example|
| :--: | :-- | :-- |
| type | Search algorithm name, including RandomSearch, AshaHpo, BohbHpo, BossHpo, and PBTHpo | `type: RandomSearch` |
| objective_keys | Optimization objective | `objective_keys:'accuracy'` |
| policy.total_epochs | Quota of epochs. Vega simplifies the configuration policy, you only need to set this parameter. For details about other parameter settings, see the examples of the HPO and NAGO algorithms. | `total_epochs: 2430` |
| tuner | Tuner type, used for the BOHB algorithm, including gp (default), rf, and hebo | tuner: "gp" |

Note: If the tuner parameter is set to hebo, the "[HEBO](https://github.com/huawei-noah/noah-research/tree/master/HEBO)" needs to be installed. Note that the gpytorch version is 1.1.1, the torch version is 1.5.0, and the torch version is 0.5.0.

Example:

```yaml
    search_algorithm:
        type: BohbHpo
        policy:
            total_epochs: 2430
```

### 3.2 Search Space

### 3.2.1 Hyperparameter Types and Constraints

The types of hyperparameters that make up the search space are as follows:

| Hyperparameter type | Example | Description |
| :--: | :-- | :-- |
| CATEGORY | `[18, 34, 50, 101]` <br> `[0.3, 0.7, 0.9]` <br> `["red", "yellow"]` <br> `[[1, 0, 1], [0, 0, 1]]` | group type. Its elements can be any data type. |
| BOOL | `[True, False]` | Boolean type |
| INT | `[10, 100]` | Integer type. Set the minimum and maximum values for even sampling. |
| INT_EXP | `[1, 100000]` | Integer type, minimum and maximum values, exponential sampling |
| FLOAT | `[0.1, 0.9]` | floating-point number type. Set the minimum and maximum values to sample evenly. |
| FLOAT_EXP | `[0.1, 100000.0]` | floating point number type. Sets the minimum and maximum values, and performs exponential sampling. |

Constraints between hyperparameters are classified into condition and forbidden, as shown in the following figure.

| Category | Constraint Type | Example | Description |
| :--: | :-- | :-- | :-- |
| condition | EQUAL | `parent: trainer.optimizer.type` <br> `child: trainer.optimizer.params.momentum` <br> `type: EQUAL` <br> `range: ["SGD"]` | indicates the relationship between two hyperparameters. The child parameter takes effect only when the parent parameter is equal to a certain value. In the example, when the value of `trainer.optimizer.type` is `["SGD"]`, the `trainer.optimizer.params.momentum` parameter takes effect. |
| condition | NOT_EQUAL | - | Indicates the relationship between two nodes. The child node takes effect only when the value of parent is different from a value. |
| condition | IN | - | Indicates the relationship between two nodes. The child node takes effect only when the parent value is within a certain range. |
| forbidden | - | - | indicates the exclusive relationship between two hyperparameter values. The two hyperparameter values cannot be used at the same time. |

The following is an example:

```yaml
hyperparameters:
    -   key: dataset.batch_size
        type: CATEGORY
        range: [8, 16, 32, 64, 128, 256]
    -   key: trainer.optimizer.params.lr
        type: FLOAT_EXP
        range: [0.00001, 0.1]
    -   key: trainer.optimizer.type
        type: CATEGORY
        range: ['Adam', 'SGD']
    -   key: trainer.optimizer.params.momentum
        type: FLOAT
        range: [0.0, 0.99]
condition:
    -   key: condition_for_sgd_momentum
        child: trainer.optimizer.params.momentum
        parent: trainer.optimizer.type
        type: EQUAL
        range: ["SGD"]
forbidden:
    -   trainer.optimizer.params.lr: 0.025
        trainer.optimizer.params.momentum: 0.35
```

In the preceding example, the forbidden configuration item is used to display the format of the forbidden configuration item.

### 3.2.2 NAS Search Space Hyperparameters

The search items in the network search space are as follows:

| Network | module | Hyperparameter | Description |
| :--: | :-- | :-- | :-- |
| ResNet | backbone | `network.backbone.depth` | Network Depth |
| ResNet | backbone | `network.backbone.base_channel` | Input Channels |
| ResNet | backbone | `network.backbone.doublechannel` | Upgrade Channel Position |
| ResNet | backbone | `network.backbone.downsample` | Downsampling Position |

The following figure shows the network configuration information, corresponding to the `model` section in the example.

| module | network | Description | Reference |
| :--: | :-- | :-- | :-- |
| backbone | ResNet | ResNet network, which consists of RestNetGeneral and LinearClassificationHead. |
| backbone | ResNetGeneral | ResNet Backbone. |
| head | LinearClassificationHead | | Network classification layer used for classification tasks. |

The following is an example in the configuration file:

```yaml
search_space:
    hyperparameters:
        -   key: network.backbone.depth
            type: CATEGORY
            range: [18, 34, 50, 101]
        -   key: network.backbone.base_channel
            type: CATEGORY
            range:  [32, 48, 56, 64]
        -   key: network.backbone.doublechannel
            type: CATEGORY
            range: [3, 4]
        -   key: network.backbone.downsample
            type: CATEGORY
            range: [3, 4]
model:
    model_desc:
        modules: ['backbone']
        backbone:
            type: ResNet
```

Other network search space configurations are determined by each algorithm. For details, see the following algorithm documents:

<table>
  <tr><th>Task</th><th>categorize</th><th>Algorithms</th></tr>
  <tr><td rowspan="3">Image Classification</td><td>Network Architecture Search</td><td><a href="../algorithms/cars.md">CARS</a>, <a href="../algorithms/nago.md">NAGO</a>, BackboneNas, DartsCNN, GDAS, EfficientNet</td></tr>
  <tr><td> Hyperparameter Optimization</td><td><a href="../algorithms/hpo.md">ASHA, BOHB, BOSS, BO, TPE, Random, Random-Pareto</a></td></tr>
  <tr><td>Data Augmentation</td><td><a href="../algorithms/pba.md">PBA</a></td></tr>
  <tr><td rowspan="2">Model Compression</td><td>Model Pruning</td><td><a href="../algorithms/prune_ea.md">Prune-EA</a></td></tr>
  <tr><td>Model Quantization</td><td><a href="../algorithms/quant_ea.md">Quant-EA</a></td></tr>
  <tr><td rowspan="2">Image Super-Resolution</td><td>Network Architecture Search</td><td><a href="../algorithms/sr_ea.md">SR-EA</a>, <a href="../algorithms/esr_ea.md">ESR-EA</a></td></tr>
  <tr><td>Data Augmentation</td><td><a href="../algorithms/cyclesr.md">CycleSR</a></td></tr>
  <tr><td>Image Segmentation</td><td>Network Architecture Search</td><td><a href="../algorithms/adelaide_ea.md">Adelaide-EA</a></td></tr>
  <tr><td>Object Detection</td><td>Network Architecture Search</td><td><a href="../algorithms/sp_nas.md">SP-NAS</a></td></tr>
  <tr><td>Lane Detection</td><td>Network Architecture Search</td><td><a href="../algorithms/auto_lane.md">Auto-Lane</a></td></tr>
  <tr><td rowspan="2">Recommender System</td><td>Feature Selection</td><td><a href="../algorithms/autofis.md">AutoFIS</a></td></tr>
  <tr><td>Feature Interactions Selection</td><td><a href="../algorithms/autogroup.md">AutoGroup</a></td></tr>
</table>

### 3.2.3 HPO Search Space Hyperparameters

Network training hyperparameters include the following:

1. Dataset parameters.
2. Model trainer parameters, including:
   1. Optimizationer and parameters.
   2. Learning rate scheduler and its parameters.
   3. Loss function and its parameters.

Configuration item description:

| Hyperparameter | Example | Description |
| :--: | :-- | :-- |
| dataset.\<dataset param\> | `dataset.batch_size` | Dataset parameter |
| trainer.optimizer.type | `trainer.optimizer.type` | Optimizer type |
| trainer.optimizer.params.\<optimizer param\> | `trainer.optimizer.params.lr` <br> `trainer.optimizer.params.momentum` | Optimizer parameter |
| trainer.lr_scheduler.type | `trainer.lr_scheduler.type` | LR-Schecduler type |
| trainer.lr_scheduler.params.\<lr_scheduler param\> | `trainer.lr_scheduler.params.gamma` | LR-Scheduler parameter |
| trainer.loss.type | `trainer.loss.type` | Loss function type |
| trainer.loss.params.\<loss function param\> | `trainer.loss.params.aux_weight` | Loss function parameter |

The configuration in the preceding table is in the following format in the configuration file:

```yaml
hyperparameters:
    -   key: dataset.batch_size
        type: CATEGORY
        range: [8, 16, 32, 64, 128, 256]
    -   key: trainer.optimizer.type
        type: CATEGORY
        range: ["Adam", "SGD"]
    -   key: trainer.optimizer.params.lr
        type: FLOAT_EXP
        range: [0.00001, 0.1]
    -   key: trainer.optimizer.params.momentum
        type: FLOAT
        range: [0.0, 0.99]
    -   key: trainer.lr_scheduler.type
        type: CATEGORY
        range: ["MultiStepLR", "StepLR"]
    -   key: trainer.lr_scheduler.params.gamma
        type: FLOAT
        range: [0.1, 0.5]
    -   key: trainer.loss.type
        type: CATEGORY
        range: ["CrossEntropyLoss", "MixAuxiliaryLoss"]
    -   key: trainer.loss.params.aux_weight
        type: FLOAT
        range: [0, 1]
condition:
    -   key: condition_for_sgd_momentum
        child: trainer.optimizer.params.momentum
        parent: trainer.optimizer.type
        type: EQUAL
        range: ["SGD"]
    -   key: condition_for_MixAuxiliaryLoss_aux_weight
        child: trainer.loss.params.aux_weight
        parent: trainer.loss.type
        type: EQUAL
        range: ["MixAuxiliaryLoss"]
```

### 3.3 Hybrid Search of NAS and HPO

NAS and HPO configuration items can be configured at the same time. The network structure and training parameters can be searched at the same time. In the following example, the model training hyperparameters are batch_size, optimizer, and ResNet network parameters depth, base_channel, doublechannel, and downsample.

```yaml
search_algorithm:
    type: BohbHpo
    policy:
        total_epochs: 100
        repeat_times: 2

search_space:
    hyperparameters:
        -   key: dataset.batch_size
            type: CATEGORY
            range: [8, 16, 32, 64, 128, 256]
        -   key: trainer.optimizer.type
            type: CATEGORY
            range: ["Adam", "SGD"]
        -   key: trainer.optimizer.params.lr
            type: FLOAT_EXP
            range: [0.00001, 0.1]
        -   key: trainer.optimizer.params.momentum
            type: FLOAT
            range: [0.0, 0.99]
        -   key: network.backbone.depth
            type: CATEGORY
            range: [18, 34, 50, 101]
        -   key: network.backbone.base_channel
            type: CATEGORY
            range:  [32, 48, 56, 64]
        -   key: network.backbone.doublechannel
            type: CATEGORY
            range: [3, 4]
        -   key: network.backbone.downsample
            type: CATEGORY
            range: [3, 4]

    condition:
        -   key: condition_for_sgd_momentum
            child: trainer.optimizer.params.momentum
            parent: trainer.optimizer.type
            type: EQUAL
            range: ["SGD"]

model:
    model_desc:
        modules: ['backbone']
        backbone:
            type: ResNet
```

## 4. Data-Agumentation configuration item

Similar to HPO, data augmentation configuration items include pipe_step, search_algorithm, search_space, dataset, trainer, and evaluator. Vega provides two data augmentation algorithms: PBA and CycleSR, for details, see [PBA](../algorithms/pba.md) and [CycleSR](../algorithms/cyclesr.md) .

## 5. Fully Train Configuration

The network model and training hyperparameter obtained after the NAS and HPO are used as the input of the Fully Train step. The fully trained model is obtained after the Fully Train step. The configuration items are as follows:

The HPO/NAS configuration items are as follows:

| Configuration item | Description |
| :--: | :-- |
| pipe_step / type | Set this parameter to `TrainPipeStep`, indicating that this step is a search step. |
| pipe_step / models_folder | Specify the location of the model description file. Read the model description files named `desc_<ID>.json` (ID indicates a number) in the folder and train these models in sequence. This option takes precedence over the model option. |
| model / model_desc_file | Location of the model description file. The priority of this configuration item is lower than that of `pipe_step/models_folder` and higher than that of `model/model_desc`. |
| model / model_desc | Model description. For details, see the model-related section in the search space. This configuration has a lower priority than `pipe_step/models_folder` and `model/model_desc`. |
| dataset | Dataset configuration. For details, see the dataset section in this document. |
| trainer | Model training parameter configuration. For details, see the trainer section in this document. |
| evaluator | evaluator parameter configuration. For details, see the evaluator section in this document. |

```yaml
my_fully_train:
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/nas/"
    trainer:
        <trainer params>
    model:
            <model desc params>
        model_desc_file: "./desc_0.json"
    dataset:
        <dataset params>
    trainer:
        <trainer params>
    evaluator:
        <evaluator params>
```

## 6. Trainer configuration item

The configuration items of the Trainer are as follows:

| Configuration item | Default value | Description |
| :--: | :-- | :-- |
| type | "Trainer" | Type |
| epochs | 1 | Number of epochs |
| distributed | False | Whether to enable horovod. To enable Horovod, set shuffle of the dataset to False. |
| syncbn | False | Whether to enable SyncBN |
| amp | False | Whether to enable the AMP |
| optimizer/type | "Adam" | Optimizer name |
| optimizer/params | {"lr": 0.1} | Optimizer Parameter |
| lr_scheduler/type | "MultiStepLR" | lr scheduler and Parameters |
| lr_scheduler/params | {"milestones": [75, 150], "gamma": 0.5} | lr scheduler and Parameters |
| loss/type | "CrossEntropyLoss" | loss and Parameters |
| loss/params | {} | loss and parameters |
| metric/type | "accuracy" | metric and parameter |
| metric/params | {"topk": [1, 5]} | metric and Parameters |
| report_freq | 10 | Frequency for printing epoch information |

Complete configuration example:

```yaml
my_fullytrain:
    pipe_step:
        type: TrainPipeStep
        # models_folder: "{local_base_path}/output/nas/"
    trainer:
        ref: nas.trainer
        epochs: 160
        optimizer:
            type: SGD
            params:
                lr: 0.1
                momentum: 0.9
                weight_decay: 0.0001
        lr_scheduler:
            type: MultiStepLR
            params:
                milestones: [60, 120]
                gamma: 0.5
    model:
        model_desc:
            modules: ['backbone']
            backbone:
                type: ResNet
        # model_desc_file: "./desc_0.json"
    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
```

In addition, Vega provides the ScriptRunner for running user scripts.

| Configuration Item | Value | Example |
| :--: | :-- | :-- |
| type | "ScriptRunner" | type: "ScriptRunner" |
| script | Script file name | "./train.py" |

For details, see the [example](https://github.com/huawei-noah/vega/blob/master/vega/examples/features/script_runner) of the trainer.

## 8. Dataset Reference

Vega provides multiple dataset classes for reading common research datasets and provides common dataset operation methods. The dataset classes provided by Vega can be configured separately for train, val, and test. You can also configure the configuration items on the common node to take effect on the three types of data. The following is a configuration example of the Cifar10 dataset:

```yaml
dataset:
    type: Cifar10
    common:
        data_path: /cache/datasets/cifar10
        batch_size: 256
    train:
        shuffle: True
    val:
        shuffle: False
    test:
        shuffle: False
```

The following describes the configuration of common data classes:

### 8.1 Cifar10 and Cifar100

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| data_path | ~ | Directory generated after the dataset is downloaded and decompressed. |
| batch_size | 256 | batch size |
| shuffle | False | shuffle |
| num_workers | 8 | Number of read threads |
| pin_memory | True | Pin memeory |
| drop_laster | True | Drop last |
| distributed | False | Data distribution |
| train_portion | 1 | Division ratio of the training set in the dataset |
| transforms | train: [RandomCrop, RandomHorizontalFlip, ToTensor, Normalize] <br> val: [ToTensor, Normalize] <br> test: [ToTensor, Normalize] | 缺省transforms |

### 8.2 ImageNet

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| data_path | ~ | Directory generated after the dataset is downloaded and decompressed. |
| batch_size | 64 | batch size |
| shuffle | train: True <br> val: False <br> test: False | shuffle |
| n_class | 1000 | Category |
| num_workers | 8 | Number of read threads |
| pin_memory | True | Pin memeory |
| drop_laster | True | Drop last |
| distributed | False | Data distribution |
| train_portion | 1 | Division ratio of the training set in the dataset |
| transforms | train: [RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize] <br> val: [Resize, CenterCrop, ToTensor, Normalize] <br> test: [Resize, CenterCrop, ToTensor, Normalize] | 缺省transforms |

### 8.3 Cityscapes

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| root_path | ~ | Directory generated after the dataset is downloaded and decompressed. |
| list_file | train: train.txt <br> val: val.txt <br> test: test.txt | Index File |
| batch_size | 1 | batch size |
| num_workers | 8 | Number of read threads |
| shuffle | False | shuffle |

### 8.4 DIV2K

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| root_HR | ~ | Directory where the HR image is located. |
| root_LR | ~ | Directory where LR images are stored. |
| batch_size | 1 | batch size |
| shuffle | False | shuffle |
| num_workers | 4 | Number of read threads |
| pin_memory | True | Pin memeory |
| value_div | 1.0 | Value div |
| upscale | 2 | Up scale |
| crop | ~ | crop size of lr image |
| hflip | False | flip image horizontally |
| vflip | False | flip image vertically |
| rot90 | False | flip image diagonally |

### 8.5 AutoLane

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| data_path | ~ | Directory generated after the dataset is downloaded and decompressed. |
| batch_size | 24 | batch size |
| shuffle | False | shuffle |
| num_workers | 8 | Number of read threads |
| network_input_width | 512 | Network inpurt width |
| network_input_height | 288 | Network input height |
| gt_len | 145 | - |
| gt_num | 576 | - |
| random_sample | True | Random sample |
| transforms | [ToTensor, Normalize] | transforms |

### 8.6 Avazu

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| data_path | ~ | Directory generated after the dataset is downloaded and decompressed. |
| batch_size | 2000 | batch size |

### 8.7 ClassificationDataset

This dataset is used to read user classification data. The user dataset directory contains three subfolders: train, val, and test. The three subfolders contain the image classification tag folder, which stores images belonging to the category.

The configuration items are as follows:

| Configuration item | Default value | Description |
| :-- | :-- | :-- |
| data_path | ~ | Directory generated after the dataset is downloaded and decompressed. |
| batch_size | 1 | batch size |
| shuffle | train: True <br> val: True <br> test: False | shuffle |
| num_workers | 8 | Number of read threads |
| pin_memory | True | Pin memeory |
| drop_laster | True | Drop last |
| distributed | False | Data distribution |
| train_portion | 1 | Division ratio of the training set in the dataset |
| n_class | - | number of clases |
| cached | True | Whether to cache all data to the memory. |
| transforms | [] | transforms |
