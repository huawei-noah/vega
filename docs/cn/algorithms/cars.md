# CARS: Continuous Evolution for Efficient Neural Architecture Search

## 1. 算法介绍

在不同的应用场景中，计算资源约束条件有所不同，很多现有的NAS方法一次搜索只能得到一个网络结构，无法满足差异化的约束条件需求。此外，尽管基于进化算法的NAS方法取得了不错的性能，但是每代样本需要重头训练来进行评估，极大影响了搜索效率。考虑到现有方法的不足，我们提出一种基于连续进化的多目标高效神经网络结构搜索方法（CARS: Continuous Evolution for Efficient Neural Architecture Search）。CARS维护一个最优模型解集，每次用解集中的模型来更新超网络中的参数。在每次进化算法迭代的过程中，子代的样本可以直接从超网络和父样本中直接继承参数，有效提高了进化效率。CARS一次搜索即可获得一系列不同大小和精度的模型，供用户根据实际应用中的资源约束来挑选相应的模型。CARS算法的详细介绍请参考 <https://arxiv.org/abs/1909.04977> 。

## 2. 算法原理

CARS算法的原理如下图所示。CARS算法维护一个超网络（SuperNet），每个子网络都是超网络的一个采样。在进化算法过程中，由于所有子网络权重与超网络共享，因此使得子代样本可以直接继承父代样本的权重，从而实现连续进化。CARS算法利用pNSGA-III算法在搜索过程中对大模型进行保护，增加搜索模型的覆盖范围。CARS算法利用帕累托前沿解集中的子网络来对超网络进行权重更新，超网络权重更新与子网络进化更新交替进行。

![framework](../../images/cars_framework.png)

### 2.1 搜索空间

#### DARTS搜索空间

pipeline中的CARS算法集成了DARTS的搜索空间，整体结构大体如下：

![darts_search_sapce](../../images/cars_darts_search_sapce.png)

DARTS搜索空间的详细介绍请参考相应的[ICLR'19文章](https://arxiv.org/abs/1806.09055)。

### 2.2 配置搜索空间

搜索阶段的搜索空间配置如下：

```yaml
    search_space:
        type: SearchSpace
        modules: ['super_network']
        super_network:
           type: CARSDartsNetwork
           stem:
               type: PreOneStem
                init_channels: 16
                stem_multi: 3
            head:
                type: LinearClassificationHead
            init_channels: 16
            num_classes: 10
            auxiliary: False
            search: True
            cells:
                modules: [
                    'normal', 'normal', 'reduce',
                    'normal', 'normal', 'reduce',
                    'normal', 'normal'
                ]
                normal:
                    type: NormalCell
                    steps: 4
                    genotype:
                      [
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 4 ],
                      ]
                    concat: [2, 3, 4, 5]
                reduce:
                    type: ReduceCell
                    steps: 4
                    genotype:
                      [
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 4 ],
                      ]
                    concat: [2, 3, 4, 5]
```

## 3. 使用指导

### 3.1 dataset配置

CARS算法默认使用CIFAR-10数据集，也可以换用其他的标准数据集或者自定义的数据集。若要使用用户自定义格式的数据集，则需要实现符合Vega要求的数据集类，具体方法可参考开发手册。

CIFAR-10数据集配置信息如下：

```yaml
    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
            num_workers: 8
            train_portion: 0.5
            drop_last: False
        train:
            shuffle: True
            batch_size: 128
        val:
            batch_size: 3500
```

### 3.2 运行环境设置

在配置文件中进行参数配置，搜索模型、训练模型可参考以下配置文件：

- `vega/examples/nas/cars/cars.yml`

配置文件在`main.py`中直接传入给pipeline，两个过程会依次进行，搜索过程会搜出位于Pareto前沿的一系列模型，训练过程会把前沿的模型训到底，得到最终的模型性能。

搜索过程的主要配置如下：

```yaml
nas:
    pipe_step:
        type: SearchPipeStep

    search_algorithm:
        type: CARSAlgorithm
        policy:
            num_individual: 128
            start_ga_epoch: 50
            ga_interval: 10
            select_method: uniform #pareto
            warmup: 50

    trainer:
        type: Trainer
        darts_template_file: "{default_darts_cifar10_template}"
        callbacks: CARSTrainerCallback
        model_statistics: False
        epochs: 500
        optim:
            type: SGD
        params:
            lr: 0.025
            momentum: 0.9
            weight_decay: !!float 3e-4
        lr_scheduler:
            type: CosineAnnealingLR
            params:
                T_max: 500
                eta_min: 0.001
        loss:
            type: CrossEntropyLoss

        grad_clip: 5.0
        seed: 10
        unrolled: True
```

Fully train阶段的配置：

```yaml
fully_train:
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/nas/"

    trainer:
        ref: nas.trainer
        callbacks: DartsFullTrainerCallback
        epochs: 600
        lr_scheduler:
            type: CosineAnnealingLR
            params:
                T_max: 600.0
                eta_min: 0
        loss:
            type: MixAuxiliaryLoss
            params:
                loss_base:
                    type: CrossEntropyLoss

            aux_weight: 0.4
        seed: 100
        drop_path_prob: 0.2

    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
            num_workers: 8
            drop_last: False
            batch_size: 96
        train:
            shuffle: True
            transforms:
                - type: RandomCrop
                  size: 32
                  padding: 4
                - type: RandomHorizontalFlip
                - type: ToTensor
                - type: Normalize
                  mean:
                      - 0.49139968
                      - 0.48215827
                      - 0.44653124
                  std:
                      - 0.24703233
                      - 0.24348505
                      - 0.26158768
                - type: Cutout
                  length: 8
        val:
            batch_size: 96
            shuffle: False
```

fully train 的 models_folder 和 model_desc_n.json 参数说明：

1. models_folder: 该设置项指定了一个目录，该目录下面需要有一个或多个model_desc_n.json的文件，其中n为序号。该设置和 model_desc_file 互斥，若两者都设置，则以 model_desc_file 优先。
2. model_desc_file：指定某个模型描述文件作为trainer的输入。

### 3.3 算法输出

算法输出：

1. nas 目录：输出相对最优的多个模型描述文件。
2. fully_train 目录：输出训练后的模型权重文件。

## 4. Benchmark

请参考 [cars.yml](https://github.com/huawei-noah/vega/blob/master/examples/nas/cars/cars.yml)。
