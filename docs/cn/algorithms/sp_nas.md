# SP-NAS (Serial-to-Parallel Backbone Search for Object Detection)

## 算法介绍

SP-NAS是面向物体检测及语义分割的高效主干网络架构搜索算法，现有的物体检测器通常采用在图像分类任务上设计和预训练的特征提取网络作为主干网络，因此我们希望通过AutoML技术和网络变形提出了一种高效、搜索空间灵活、任务导向的主干网络架构搜索方案。我们提出了一个从串行（Serial）到并行（Parallel）的两阶段搜索方案，减少重复的ImageNet预训练或者长时间的train from scratch。

## 算法原理

该方法可分为2阶段：

1. 串行阶段通过“swap-expand-reignite”搜索策略找到具有最佳缩放比例和输出通道的block序列，该搜索策略可以完全继承网络变形前的权重；
2. 并行阶段设计了全新的并行网络结构，通过搜索不同特征层次集成的子网络数，更好的融合高级语义和低级语义特征。搜索策略图如下：

![sp-nas](../../images/sp_nas.png)

## 搜索空间和搜索策略

**串行搜索阶段 Serial-level Search**

- “swap-expand-reignite”策略：从小网络开始生长，避免重复的ImageNet预训练
  - 采用多次“交换”或者“扩展”对网络变形得到新的候选网络；
  - 继承参数快速训练并评估候选网络；
  - 生长到达瓶颈时，对网络进行reignite（ ImageNet 训练），点火次数≤2。

- 限制条件的最优网络：在给定的网络资源（时延，显存占用或复杂度）限制下，挑选能够获得最大效益的串行网络。
- 搜索空间
  - Block类型：基础残差基元Basic Block, 瓶颈残差基元BottleNeck Block和ResNext基元；
  - 网络深度 (depth)：8-60个Blocks；
  - 特征层次阶段数 (stage)：5-7个阶段；
  - 网络宽度 (width)：搜索整个序列中通道大小加倍的位置。

**并行搜索阶段 Parallel-level Search**

- 基于串行搜索阶段的结果SerialNet（or 已有的串行网络如ResNet系列），搜索堆叠在SerialNet上的并行结构，旨在更好地利用和融合来自不同特征层次阶段的，具有不同分辨率的特征信息。
- 搜索策略：资源约束的随机采样，添加额外子网络的概率与待添加子网络的FLOPS成反比。
- 搜索空间：根据特征层次阶段数划分SerialNet为L个自网络，搜索每个阶段堆叠的子网络个数K。

## 使用指导

### fine tune：将torchvision的权重文件转换为spnas的权重
```yaml

fine_tune:
    pipe_step:
        type: TrainPipeStep

    model:
        pretrained_model_file: /cache/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth  # 指定权重文件路径
        model_desc:
            type: FasterRCNN
            convert_pretrained: True     # 将torchvision的预训练权重适配到backbone中
            backbone:
                type: SerialBackbone     # 指定backbone类型

```

### 阶段1：串行阶段

```yaml
    search_algorithm:
        type: SpNasS
        max_sample: 20              # 最多采用结构数
        objective_keys: ['mAP', 'params']   # 优化目标为mAP和params组成的pareto前沿 
        max_optimal: 5              # 串行阶段保留Top5种子网络开始变异，并行阶段设为1
        num_mutate: 3               # 变异次数
        add_stage_ratio: 0.05       # 新增特征层次阶段数的概率
        expend_ratio: 0.3           # 新增block数的概率
        max_stages: 6               # 最大可允许的特征层次阶段数
    
    search_space:
        type: SearchSpace
        hyperparameters:
            -   key: network.backbone.code
                type: CATEGORY
                range: ['111-2111-211111-211']

    model:
        pretrained_model_file: "{local_base_path}/output/fine_tune/model_0.pth"   # 从fine_tune中获取预训练权重
        model_desc:
            type: FasterRCNN         # 网络类型
            freeze_swap_keys: True   # 冻结没有交换的block
            backbone:                # block类型
                type: SerialBackbone
    
```

### 阶段2：重燃阶段

```yaml
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/serial/"  # 指定从阶段1获取模型和权重文件

    trainer:
        type: Trainer
        callbacks: ReignitionCallback   # 指定重燃的callback
```

### 阶段3：并行阶段

```yaml
     pipe_step:
        type: SearchPipeStep
        models_folder: "{local_base_path}/output/reignition/"  # 从阶段二获取网络信息

    search_algorithm:
        type: SpNasP
        max_sample: 1

    model:
        pretrained_model_file:  "{local_base_path}/output/fine_tune/model_0.pth"  # 加载fasterRcnn的权重
        model_desc:
            type: FasterRCNN
            neck:
              type: ParallelFPN  # neck类型

    search_space:
        type: SearchSpace
        hyperparameters:
            -   key: network.neck.code   # neck的搜索空间
                type: CATEGORY
                range: [[0, 1, 2, 3]]
```

### 阶段4：fully train

```yaml
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/parallel/"  # 从阶段3获取模型和权重信息
```

### 算法输出

- 搜索到的模型经充分训练后得到的模型及结果。
- 整个搜索过程中所有模型的结果{local_base_path}/output中。

## Benchmark

Benchmark配置信息请参考: [spnas.yml](https://github.com/huawei-noah/vega/blob/master/examples/nas/sp_nas/spnas.yml)
