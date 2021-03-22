# SR-EA

## 算法介绍

SR-EA是利用进化算法（EA）搜索图片超分（SR）网络架构的模块。EA是一种常用的自动网络架构搜索方法（NAS）。搜索的过程通常为如下的步骤：

1. 用某种方式（通常为随机）生成一系列的模型，并对每个模型进行不完全训练（如降低迭代次数，减小训练样本等）；
2. 计算出目前生成所有模型的帕雷托前沿（pareto front），以此为基础生成进化模型，并对每个模型进行不完全训练；
3. 不断重复步骤2，直到达到指定的代数或生成指定的模型数等。

## 算法原理

SR-EA目前里面提供了两种网络架构，分别为Modified SRResNet（作为baseline），以及CCRN-NAS（诺亚自研）。Modified SRResNet的结构如下图所示：

![Modified SRResNet](../../images/sr_ea_SRResNet.png)

SR-EA为Modified SRResNet提供了random search和brute force search两种架构搜索的方法，用于搜索架构中block的数量及通道数。

CCRN-NAS是一种专门用于自动架构搜索的网络架构，在轻量级网络上有较好的表现。CCRN-NAS由三种block组成：

1. Kernel size为2的residual block。
2. Kernel size为3的residual block。
3. Channel Increase Block (CIB)：顺次通过两个模块，每个模块由1或2组成，并将两个输出在通道维度合并。因此在通过CIB之后，通道数加倍。

Pipeline为CCRN-NAS提供了EA架构搜索方法，搜索3种模块的组合而对网络架构做优化。

更多的网络架构及搜索空间待开发。

## 搜索空间和搜索策略

Modified SRResNet的搜索空间包括block的数量及通道数，并提供了random和brute force两种搜索方法。两种搜索方法中，用户均定义block数量的选择，及通道数的选择。random search会在这些选择中随机生成模型训练，直到模型数达到max_count，而brute force搜索会训练选择中所有的模型。

CCRN-NAS的搜索空间为三种block的组合：

1. 随机搜索：用户定义residual block数量的选择和CIB数量的选择。Residual block的数量和CIB的数量会在用户的选择中随机产生；Residual block中，kernel size为2的比例在[0,1]中随机生成。pipeline会先生成普通的residual block，然后将CIB随机插入到residual block中。
2. 进化搜索：pipeline每次会从pareto front中随机抽取一个模型进行更改，用户自定义更改的次数。每次更改可进行如下操作：
   - 将随机一个residual block的kernel size由2改成3，或由3改成2。
   - 在随机的层数，增加一个residual block，kernel size随机在2和3中产生。

## 配置搜索空间

搜索空间和搜索算法的配置项如下：

```yaml
pipeline: [random, mutate]

random:
    pipe_step:
        type: SearchPipeStep

    search_space:
        type: SearchSpace
        modules: ['custom']
        custom:
            type: MtMSR
            in_channel: 3
            out_channel: 3
            upscale: 2
            rgb_mean: [0.4040, 0.4371, 0.4488]
            candidates: [res2, res3]
            block_range: [10, 80]
            cib_range: [3, 4]

    search_algorithm:
        type: SRRandom
        codec: SRCodec
        policy:
            mum_sample: 1000

mutate:
    search_space:
        ref: random.search_space

    search_algorithm:
        type: SRMutate
        codec: SRCodec
        policy:
            mum_sample: 1000
            num_mutate: 3
```

### 算法输出

算法的输出有

- 搜索到的帕雷托前沿的模型经充分训练后得到的模型及结果。
- 随机搜索及进化搜索过程中所有模型的结果reports.csv，以及帕雷托前沿的结果output.csv。
