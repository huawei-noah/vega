# 配置参考

`Vega`将从数据到模型的整个AutoML过程，分解为多个步骤，这些步骤包括网络架构搜索、超参优化、数据增广、模型训练等，Vega可以通过配置文件组合这些步骤成为一个完整的流水线，依次执行这些步骤，完成从数据到模型的全流程。

同时针对网络架构搜索、超参优化、数据增广等算法，Vega设计了独立于搜索算法的网络和超参搜索空间，可根据实际需要，通过调整配置文件，实现个性化的搜索。

如下是运行CARS算法示例：

```bash
cd examples
vega ./nas/cars/cars.yml
```

以下详细介绍配置文件中的每个配置项。

## 1. 整体结构

vega的配置可分为两部分：

1. 通用配置，配置项名称是`general`，用于设置公共和通用的一些配置项，如Backend、输出路径和日志级别等。
2. pipeline配置，包含两部分：
   1. pipeline定义，配置项名称是`pipeline`，是一个列表，包含了pipeline中的各个步骤。
   2. pipeline中各个步骤的定义，配置项名称是pipeline中定义的各个步骤名称。

```yaml
# 此处配置公共配置项，可参考随后的章节介绍
general:
    # general configuration

# 定义pipeline。
pipeline: [my_nas, my_hpo, my_data_augmentation, my_fully_train]

# 定义每个步骤，可参考随后的章节介绍
my_nas:
    # NAS configuration

my_hpo:
    # HPO configuration

my_data_augmentation:
    # Data augmentation configuration

my_fully_train:
    # fully train configuration
```

以下详细介绍每个配置项。

## 2. 公共配置项

公共配置项中可以配置的配置项有：

| 配置项 | 可选项 | 缺省值 | 说明 |
| :--: | :-- | :-- | :-- |
| backend | pytorch \| tensorflow \| mindspore | pytorch | 设置Backend。  |
| local_base_path | - | ./tasks/ | 工作路径。每次系统运行，会在该路径下生成一个带有时间信息（我们称之为task id）的子文件夹，这样多次运行的输出不会被覆盖。在task id子文件夹下面一般包含output和worker两个子文件夹，output文件夹存储pipeline的每个步骤的输出数据，worker文件夹保存临时信息。 <br> **在集群的场景下，该路径需要设置为每个计算节点都可访问的EFS路径，用于不同节点共享数据。** |
| timeout | - | 10 | worker超时时间，单位为小时，若在该时间范围内未完成，worker会被强制结束。 |
| parallel_search | True \| False | False | 是否并行搜索多个模型。 |
| parallel_fully_train | True \| False | False | 是否并行训练多个模型。 |
| devices_per_trainer | 1..N (N为单节点最大GPU/NPU数) | 1 | 并行搜索和训练时，每个trainer分配的设备（GPU \| NPU)数目。当parallel_search或parallel_fully_train为True时生效。缺省为1，每个trainer分配1个（GPU \| NPU）。 |
| logger / level | debug \| info \| warn \| error \| critical | info | 日志级别。 |
| cluster / master_ip | - | ~ | 在集群场景下需要设置该参数，设置为master节点的IP地址。 |
| cluster / listen_port | - | 8000 | 在集群场景下需要关注该参数，若出现8000端口被占用，需要调整该监控端口。 |
| cluster / slaves | - | [] | 在集群场景下需要设置该参数，设置为除了master节点外的其他节点的IP地址。 |
| quota / restrict / flops | - | ~ | 采样模型的浮点计算量最大值或范围，单位为M。 |
| quota / restrict / params | - | ~ | 采样模型的参数量最大值或范围，单位为K。 |
| quota / restrict / latency | - | ~ | 采样模型的时延最大值或范围，单位为ms。 |
| quota / target / type | accuracy \| IoUMetric \| PSNR | ~ | 模型的训练metric目标类型。 |
| quota / target / value | - | ~ | 模型的训练metric目标值。 |
| quota / runtime | - | ~ | 用户设定的Pipeline最大运行时间估计值，单位为h。 |

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
        listen_port: 8000
        slaves: []
    quota:
        restrict:
            flops: 10
            params: [100, 1000]
            latency: 100
        target:
            type: accuracy
            value: 0.98
        runtime: 10
```

## 2.1 并行和分布式

涉及到分布式的配置项有：parallel_search, parallel_fully_train 和 trainer.distributed，若有多张GPU|NUP，可根据需要选择合适的并行和分布式设置。

1. 在搜索阶段可考虑如下设置：

    ```yaml
    general:
        parallel_search: True
    ```

2. 在fully train阶段，若需要训练较多模型，且数据量不大，可考虑一次同时训练多个模型：

    ```yaml
    general:
        parallel_fully_train: True
    ```

3. 在fully train阶段，若需要训练单个模型，且数据量较大，需要考虑多张卡同时训练一个模型：

    ```yaml
    general:
        parallel_fully_train: False

    pipeline: [fully_train]

    fully_train:
        pipe_step:
            type: TrainPipeStep
        trainer:
            distributed: True
    ```

## 3. NAS和HPO配置项

HPO / NAS的配置项有如下几个主要部分：

| 配置项 | 说明 |
| :--: | :-- |
| pipe_step / type | 配置为`SearchPipeStep`标识本步骤为搜索步骤 |
| search_algorithm | 搜索算法配置，详见本文搜索算法章节 |
| search_space | 搜索空间配置，详见本文搜索空间章节 |
| model | 模型配置，详见本文搜索空间章节 |
| dataset | 数据集配置，详见本文数据集章节 |
| trainer | 模型训练参数配置，详见本文训练器章节 |
| evaluator | 评估器参数配置，详见本文评估器章节 |

在配置文件中，配置项如下：

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

以下详细介绍search_algorithm和search_space配置项。

### 3.1 搜索算法

一般搜索算法包括如下配置项：

| 配置项 | 说明 | 示例 |
| :--: | :-- | :-- |
| type | 搜索算法名称 | `type: BackboneNas` |
| objective_keys | 优化目标 | 指定当前算法选取的优化目标，对应trainer配置的metrics，trainer默认metrics额外提供`flops`和`params`两个metrics<br/>如果配置了evaluator，额外增加`latency`作为metrics<br/>如果是单目标优化问题，指定优化目标名称：`objective_keys: 'accuracy'`，<br/>如果是多目标优化问题，采用数组形式表示：`objective_keys: ['accuracy', 'flops','latency']`，<br/>系统默认配置为`objective_keys:  'accuracy'` <br/>|
| policy | 搜索策略，搜素算法自身参数 | 比如BackboneNas使用进化算法，其策略配置为： <br> `num_mutate: 10` <br> `random_ratio: 0.2` |
| range | 搜索范围 | 比如BackboneNas的搜索范围可以确定为：<br> `min_sample: 10` <br> `max_sample: 300` |

上表中搜索算法的示例在配置文件中为：

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

以上是以搜索算法BackboneNas为例，不同搜索算法有不同的配置项，请参考各个搜索算法文档的相关章节。

<table>
  <tr><th>任务</th><th>分类</th><th>参考算法</th></tr>
  <tr><td rowspan="3">图像分类</td><td>网络架构搜索</td><td><a href="../algorithms/cars.md">CARS</a>、DartsCNN、GDAS、BackboneNas、EfficientNet</td></tr>
  <tr><td>超参优化</td><td><a href="../algorithms/hpo.md">ASHA、BOHB、BOSS、BO、TPE、Random、Random-Pareto</a></td></tr>
  <tr><td>数据增广</td><td><a href="../algorithms/pba.md">PBA</a></td></tr>
  <tr><td rowspan="2">模型压缩</td><td>模型剪枝</td><td><a href="../algorithms/prune_ea.md">Prune-EA</a></td></tr>
  <tr><td>模型量化</td><td><a href="../algorithms/quant_ea.md">Quant-EA</a></td></tr>
  <tr><td rowspan="2">图像超分辨率</td><td>网络架构搜索</td><td><a href="../algorithms/sr_ea.md">SR-EA</a>、<a href="../algorithms/esr_ea.md">ESR-EA</a></td></tr>
  <tr><td>数据增广</td><td><a href="../algorithms/cyclesr.md">CycleSR</a></td></tr>
  <tr><td>图像语义分割</td><td>网络架构搜索</td><td><a href="../algorithms/adelaide_ea.md">Adelaide-EA</a></td></tr>
  <tr><td>物体检测</td><td>网络架构搜索</td><td><a href="../algorithms/sp_nas.md">SP-NAS</a></td></tr>
  <tr><td>车道线检测</td><td>网络架构搜索</td><td><a href="../algorithms/auto_lane.md">Auto-Lane</a></td></tr>
  <tr><td rowspan="2">推荐搜索</td><td>特征选择</td><td><a href="../algorithms/autofis.md">AutoFIS</a></td></tr>
  <tr><td>特征交互建模</td><td><a href="../algorithms/autogroup.md">AutoGroup</a></td></tr>
</table>

### 3.2 搜索空间

### 3.2.1 超参类型和约束

组成搜索空间的超参的类型如下：

| 超参类型 | 示例 | 说明 |
| :--: | :-- | :-- |
| CATEGORY | `[18, 34, 50, 101]` <br> `[0.3, 0.7, 0.9]` <br> `["red", "yellow"]` <br> `[[1, 0, 1], [0, 0, 1]]` | 分组类型，其元素可以为任意的数据类型 |
| BOOL | `[True, False]` | 布尔类型 |
| INT | `[10, 100]`  | 整数类型，设置最小值和最大值，均匀采样 |
| INT_EXP | `[1, 100000]` | 整数类型，设置最小值和最大值，指数采样 |
| FLOAT | `[0.1, 0.9]`  | 浮点数类型，设置最小值和最大值，均匀采样 |
| FLOAT_EXP | `[0.1, 100000.0]` | 浮点数类型，设置最小值和最大值，指数采样 |

超参间的约束分为condition和forbidden，如下：

| 类别 | 约束类型 | 示例 | 说明 |
| :--: | :-- | :-- | :-- |
| condition | EQUAL | `parent: trainer.optimizer.type` <br> `child: trainer.optimizer.params.momentum` <br> `type: EQUAL` <br> `range: ["SGD"]` | 用于表示2个超参之间的关系，当parent参数等于某一个值时，child参数才会生效。如示例中，当`trainer.optimizer.type` 的取值为 `["SGD"]` 时，参数 `trainer.optimizer.params.momentum` 生效。 |
| condition | NOT_EQUAL | - | 用于表示2个节点之间的关系，当parent不等于某一个值时，child节点才会生效。 |
| condition | IN | - | 用于表示2个节点之间的关系，当parent在一定范围内时，child节点才会生效。 |
| forbidden | - | - | 用于表示2个超参值的互斥关系，配置中的两个超参值不能同时出现。 |

示例如下：

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

上例中forbidden的配置用于展示forbidden配置项的格式。

### 3.2.2 NAS搜索空间超参

网络搜索空间的可搜索项有：

| 网络 | module | 超参 | 说明 |
| :--: | :-- | :-- | :-- |
| ResNet | backbone | `network.backbone.depth` | 网络深度 |
| ResNet | backbone | `network.backbone.base_channel` | 输入channel数 |
| ResNet | backbone | `network.backbone.doublechannel` | 升通道位置 |
| ResNet | backbone | `network.backbone.downsample` | 下采样位置 |

构建网络配置信息如下，对应示例中的`model`节：

| module | network | 说明 | 参考 |
| :--: | :-- | :-- | :-- |
| backbone | ResNet | ResNet网络，由RestNetGeneral和LinearClassificationHead组成。 |
| backbone | ResNetGeneral | ResNet网络Backbone。 |
| head | LinearClassificationHead | | 用于分类任务的网络分类层。 |

上表中的示例，在配置文件中如下：

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

其他的网络搜索空间配置，由各个算法确定，请参考各个算法文档：

<table>
  <tr><th>任务</th><th>分类</th><th>参考算法</th></tr>
  <tr><td rowspan="3">图像分类</td><td>网络架构搜索</td><td><a href="../algorithms/cars.md">CARS</a>、DartsCNN、GDAS、BackboneNas、EfficientNet</td></tr>
  <tr><td>超参优化</td><td><a href="../algorithms/hpo.md">ASHA、BOHB、BOSS、BO、TPE、Random、Random-Pareto</a></td></tr>
  <tr><td>数据增广</td><td><a href="../algorithms/pba.md">PBA</a></td></tr>
  <tr><td rowspan="2">模型压缩</td><td>模型剪枝</td><td><a href="../algorithms/prune_ea.md">Prune-EA</a></td></tr>
  <tr><td>模型量化</td><td><a href="../algorithms/quant_ea.md">Quant-EA</a></td></tr>
  <tr><td rowspan="2">图像超分辨率</td><td>网络架构搜索</td><td><a href="../algorithms/sr_ea.md">SR-EA</a>、<a href="../algorithms/esr_ea.md">ESR-EA</a></td></tr>
  <tr><td>数据增广</td><td><a href="../algorithms/cyclesr.md">CycleSR</a></td></tr>
  <tr><td>图像语义分割</td><td>网络架构搜索</td><td><a href="../algorithms/adelaide_ea.md">Adelaide-EA</a></td></tr>
  <tr><td>物体检测</td><td>网络架构搜索</td><td><a href="../algorithms/sp_nas.md">SP-NAS</a></td></tr>
  <tr><td>车道线检测</td><td>网络架构搜索</td><td><a href="../algorithms/auto_lane.md">Auto-Lane</a></td></tr>
  <tr><td rowspan="2">推荐搜索</td><td>特征选择</td><td><a href="../algorithms/autofis.md">AutoFIS</a></td></tr>
  <tr><td>特征交互建模</td><td><a href="../algorithms/autogroup.md">AutoGroup</a></td></tr>
</table>

### 3.2.3 HPO 搜索空间超参

网络训练超参包括如下超参：

1. 数据集参数。
2. 模型训练器参数，包括：
   1. 优化方法，及其参数。
   2. 学习率策略，及其参数。
   3. 损失函数，及其参数。

配置项说明：

| 超参 | 示例 | 说明 |
| :--: | :-- | :-- |
| dataset.\<dataset param\> | `dataset.batch_size` | 数据集参数 |
| trainer.optimizer.type | `trainer.optimizer.type` | 优化器类型 |
| trainer.optimizer.params.\<optimizer param\> | `trainer.optimizer.params.lr` <br> `trainer.optimizer.params.momentum` | 优化器参数 |
| trainer.lr_scheduler.type | `trainer.lr_scheduler.type` | 学习率策略类型 |
| trainer.lr_scheduler.params.\<lr_scheduler param\> | `trainer.lr_scheduler.params.gamma` | 学习率策略参数 |
| trainer.loss.type | `trainer.loss.type` | 损失函数类型 |
| trainer.loss.params.\<loss function param\> | `trainer.loss.params.aux_weight` | 损失函数参数 |

如上表格示例中的配置在配置文件中格式如下：

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

### 3.3 NAS 和 HPO 混合搜索

NAS 和 HPO 的配置项可以同时配置，同时搜索网络结构和训练参数，如下例，同时搜索的模型训练超参数为batch_size, optimizer，及ResNet的网络参数depth，base_channel，doublechannel，downsample：

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

## 4. Data-Agumentation 配置项

数据增广的配置项类似于HPO，有pipe_step、search_algorithm、search_space、dataset、trainer、evaluator配置项，Vega提供了两个数据增广的算法：PBA和CycleSR，请参考这两个算法文档：[PBA](../algorithms/pba.md)，[CycleSR](../algorithms/cyclesr.md)。

## 5. Fully Train配置项

经过NAS、HPO后，得到的网络模型和训练超参，可以作为Fully Train这个步骤的输入，经过Fully Train，得到完全训练后的模型，其配置项如下：

HPO/NAS的配置项有如下几个主要部分：

| 配置项 | 说明 |
| :--: | :-- |
| pipe_step / type | 配置为`TrainPipeStep`标识本步骤为搜索步骤 |
| pipe_step / models_folder | 指定模型描述文件所在的位置，依次读取该文件夹下文件名为`desc_<ID>.json`(其中ID为数字)的模型描述文件，依次训练这些模型。这个选项优先于model选项。 |
| model / model_desc_file | 模型描述文件位置。该配置项优先级低于 `pipe_step/models_folder` ，高于 `model/model_desc`。 |
| model / model_desc | 模型描述，详见本文搜索空间中model相关章节。该配置优先级低于 `pipe_step/models_folder` 和 `model/model_desc` |
| dataset | 数据集配置，详见本文数据集章节 |
| trainer | 模型训练参数配置，详见本文训练器章节 |
| evaluator | 评估器参数配置，详见本文评估器章节 |

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

## 6. Trainer 配置项

Trainer的配置项如下：

| 配置项 | 缺省值 | 说明 |
| :--: | :-- | :-- |
| type | "Trainer" | 类型 |
| epochs | 1 | epochs数 |
| distributed | False | 是否启用horovod。启用Horovod需要将数据集的shuffle参数设置为False。 |
| syncbn | False | 是否启用SyncBN |
| amp | False | 是否启用AMP |
| optimizer/type | "Adam" | 优化器名称 |
| optimizer/params | {"lr": 0.1} | 优化器参数 |
| lr_scheduler/type | "MultiStepLR" | lr scheduler 及参数 |
| lr_scheduler/params | {"milestones": [75, 150], "gamma": 0.5} | lr scheduler 及参数 |
| loss/type | "CrossEntropyLoss" | loss 及参数 |
| loss/params | {} | loss 及参数 |
| metric/type | "accuracy" | metric 及参数 |
| metric/params | {"topk": [1, 5]} | metric 及参数 |
| report_freq | 10 | 打印epoch信息的频率 |

完整配置示例：

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

## 8. 数据集参考

Vega提供了多种数据集类用于读取常用的研究用数据集，并提供了常用的数据集操作方法。Vega提供的数据集类可将train、val、test分开配置，也可以将配置项配置在common节点，同时作用到这三类数据。以下为Cifar10数据集的配置示例：

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

以下介绍常用的数据类的配置：

### 8.1 Cifar10 和 Cifar100

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| data_path | ~ | 下载数据集后，解压缩后的目录。 |
| batch_size | 256 | batch size |
| shuffle | False | shuffle |
| num_workers | 8 | 读取线程数 |
| pin_memory | True | Pin memeory |
| drop_laster | True | Drop last |
| distributed | False | 数据分布 |
| train_portion | 1 | 数据集中训练集的划分比例 |
| transforms | train: [RandomCrop, RandomHorizontalFlip, ToTensor, Normalize] <br> val: [ToTensor, Normalize] <br> test: [ToTensor, Normalize] | 缺省transforms |

### 8.2 ImageNet

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| data_path | ~ | 下载数据集后，解压缩后的目录。 |
| batch_size | 64 | batch size |
| shuffle | train: True <br> val: False <br> test: False | shuffle |
| n_class | 1000 | 分类 |
| num_workers | 8 | 读取线程数 |
| pin_memory | True | Pin memeory |
| drop_laster | True | Drop last |
| distributed | False | 数据分布 |
| train_portion | 1 | 数据集中训练集的划分比例 |
| transforms | train: [RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize] <br> val: [Resize, CenterCrop, ToTensor, Normalize] <br> test: [Resize, CenterCrop, ToTensor, Normalize] | 缺省transforms |

### 8.3 Cityscapes

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| root_path | ~ | 下载数据集后，解压缩后的目录。 |
| list_file | train: train.txt <br> val: val.txt <br> test: test.txt | 索引文件 |
| batch_size | 1 | batch size |
| num_workers | 8 | 读取线程数 |
| shuffle | False | shuffle |

### 8.4 DIV2K

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| root_HR | ~ | HR图片所在的目录。 |
| root_LR | ~ | LR图片所在的目录。 |
| batch_size | 1 | batch size |
| shuffle | False | shuffle |
| num_workers | 4 | 读取线程数 |
| pin_memory | True | Pin memeory |
| value_div | 1.0 | Value div |
| upscale | 2 | Up scale |
| crop | ~ | crop size of lr image |
| hflip | False | flip image horizontally |
| vflip | False | flip image vertically |
| rot90 | False | flip image diagonally |

### 8.5 AutoLane

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| data_path | ~ | 下载数据集后，解压缩后的目录。 |
| batch_size | 24 | batch size |
| shuffle | False | shuffle |
| num_workers | 8 | 读取线程数 |
| network_input_width | 512 | Network inpurt width |
| network_input_height | 288 | Network input height |
| gt_len | 145 | - |
| gt_num | 576 | - |
| random_sample | True | Random sample |
| transforms | [ToTensor, Normalize] | transforms |

### 8.6 Avazu

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| data_path | ~ | 下载数据集后，解压缩后的目录。 |
| batch_size | 2000 | batch size |

### 8.7 ClassificationDataset

该数据集用于用户分类数据的读取，用户数据集目录下面应该包含三个子文件夹：train、val、test，这三个文件夹下面是图片分类标签文件夹，标签文件夹里面放置属于该分类的图片。

配置项如下：

| 配置项 | 缺省值 | 说明 |
| :-- | :-- | :-- |
| data_path | ~ | 下载数据集后，解压缩后的目录。 |
| batch_size | 1 | batch size |
| shuffle | train: True <br> val: True <br> test: False | shuffle |
| num_workers | 8 | 读取线程数 |
| pin_memory | True | Pin memeory |
| drop_laster | True | Drop last |
| distributed | False | 数据分布 |
| train_portion | 1 | 数据集中训练集的划分比例 |
| n_class | ~ | number of clases |
| cached | True | 是否将全部数据缓存到内存。 |
| transforms | [] | transforms |
