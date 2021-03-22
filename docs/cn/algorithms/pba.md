# PBA

## 1. 算法介绍

PBA是基于PBT优化算法搜索数据增广策略时刻表的一种算法，PBT是一种通过最优化性能来同时优化网络参数和超参数的超参数搜索算法。PBT算法的最终输出为一个经过训练的模型以及整个搜索过程中所得到的使性能最佳的超参数设定时刻表。在PBA算法中，我们只关心得到的数据增广策略的策略时刻表，并将其可以用于在同一个数据集上对不同模型进行训练，获得更优性能。

PBA搜索过程通常为如下步骤：

1. 选择一个固定数量的模型种群（例如16），随机初始化这些模型同时并行的对这些模型进行训练；
2. 每经过一定的训练epoch，使用“搜索-利用”算法对这个模型种群中表现最差的若干模型进行更新，即将表现最好的相同数量的模型的参数以及超参数赋予表现最差的模型，并对超参数施加一个小的扰动；
3. 不断重复步骤2，直到达到指定的epoch轮数。

当前PBA算法支持图像分类场景下的自动数据增广，可以得到与具体模型无关，仅针对当前数据集的一套数据增广策略时刻表，可用于不同模型在当前数据集上的训练。

当前PBA算法可应用于图像分类任务的数据集，使用时需要用户以vega pipeline中的Dataset基类作为父类定义自己的数据集的子类，并在配置文件中进行相关设置

## 2. 算法原理

数据增广技术在训练神经网络模型时有着广泛的应用，是提高图像分类器准确性的有效技术。但是过往的大多数据增强技术实现是手动设计的。Google最早提出了AutoAugment来自动搜索改进数据增强。在AutoAugment中，作者设计了一个搜索空间，其中一个策略由许多子策略组成，其中一个的子策略是为每个批量中的单个图像随机选择的。子策略由两个操作组成，每个操作是图像处理功能，例如平移，旋转或剪切。使用搜索算法来找到最佳策略，使得神经网络在目标数据集上产生最高的验证准确度。

针对传统的AutoAugment算法耗时长，算力要求大的特点，伯克利大学的研究团队提出了基于种群的数据增强策略（PBA）。PBA算法基于PBT算法。与AutoAugment不同，作者采取的搜索空间是连续的，并且搜索得到的是一个随着epoch不同而改变的数据增强计划表。在不同的epoch，这个算法采取不同的数据增广策略。文章作者表示，尽管最终的搜索空间比AutoAugment要大的很多。但是最终结果表明数据增广表更加适用。首先，如果不完整训练完一个子模型，我们就很难评估一个固定策略。其次，有理由相信一个数据增广策略表可以更好的搜索出来，因为数据增广表有着较好的光滑性质。

基于种群搜索算法如下。在一开始，我们将所有线程都随机设定初始值，并且并行运行。我们这里用“搜索-利用”算法来进行线程更新。我们这里每一次都把最好的线程克隆给最差的线程，然后对最好的线程的当前超参进行一些随机扰动。因为最差的线程被最好的替代了，而最好的线程被局部扰动。所以我们可以认为这个操作是进行了一次“搜索-利用”算法。

应用在PBA中，我们以CIFAR10为例，算法在CIFAR10中抽取一个4000张样本的子数据集用于数据增广策略表的搜索，并同时使用16个模型进行训练。一开始每一个线程都有一个自己的初始数据增强策略（这里类似AutoAugment算法，策略为（A,P,M）的形式，即（所选策略，使用概率，使用强度）三元组。然后每隔3个epoch都将训练后的模型在验证数据集中进行测试，得到一个val_acc，即验证准确率，之后根绝验证准确率的结果对模型进行排序，用性能最好的四个模型覆盖掉性能最差的四个模型，并分别加以一个随机扰动，持续这个过程200个epoch。

由于PBA算法一共只完整训练了16个模型，每个200个epoch，相比AutoAugment所需搜索的16000个模型，每个120个epoch，在时间效率上有了极大提升的同时，获得与AutoAugment几乎相当的实验精度。下图分别展示了PBA算法的算法流程、数据增广操作选择策略以及“搜搜-利用“算法的策略。

<center>
<img src="../../images/pba_1.png"/>
</center>

<center>
<img src="../../images/pba_2.png" width=512 height=512 />
</center>

<center>
<img src="../../images/pba_3.png" width=512 height=512 />
</center>

## 3. 搜索空间和搜索策略

当前PBA默认支持数据增广操作为15个，根据算法，对每个batch的训练样本的数据增广操作选择时，允许每个操作重复一次，且每个操作有使用概率和使用强度两个参数，因此搜索空间定义为一个60维的整数向量。

搜索方法即如算法原理中所介绍，采用PBT算法进行搜索。

## 4. 配置搜索空间

当前PBA算法支持的数据增广操作有以下15种：

<center>
<img src="../../images/pba_4.png"/>
</center>

在配置文件中，可以调整使用的数据增广操作。

```yaml
pipeline: [hpo]

hpo:
    pipe_step:
        type: SearchPipeStep
    dataset:
        type: Cifar10
    search_space:
        type: SearchSpace
        transformers:
            - Cutout: True
            - Rotate: True
            - Translate_X: True
            - Translate_Y: True
            - Brightness: True
            - Color: True
            - Invert: True
            - Sharpness: True
            - Posterize: True
            - Shear_X: True
            - Solarize: True
            - Shear_Y: True
            - Equalize: True
            - AutoContrast: True
            - Contrast: True

    search_algorithm:
        type: PBAHpo
        policy:
            each_epochs: 3            # 每轮trainer需要训练的epoch数目
            config_count: 16          # 搜索算法并行训练的模型组数
            total_rungs: 200          # 搜索算法迭代轮数
    trainer:
        type: Trainer
    evaluator:
        type: Evaluator
        host_evaluator:
            type: HostEvaluator
            metric:
                type: accuracy
```

## 5. 结果输出

PBA算法在vega pipeline上使用参数配置文件中的默认参数(搜索阶段运行200个epoch，完全训练阶段运行400个epoch)能够在CIFAR10数据集中达到与算法原论文相当的精度（所用模型为wide-resnet-28-10），如下表所示：

||Baseline|Cutout|AA|PBA|
|:--:|:--:|:--:|:--:|:--:|
|Ho et at.,2019|96.13%|96.92%|97.32%|97.42%|
|Vega Pipeline|96.26%|97.18%| \ |97.57%|

最终输出文件和目录如下：

```text
output:
    best_hps.json:    其中为pba算法搜索得到的最佳数据增广策略表及其搜索阶段的ID与得分
    hps.csv:          其中为pba算法搜索阶段得到的16组数据增广策略表的ID与得分
    score_board.csv:  其中为pba算法搜索阶段得到的16组数据增广操作每轮迭代过程中的具体得分与状态
workers:
    hpo:             其中16个文件夹分别为16组模型的最终结果，包括得分与模型等
        0:
        1:
        ...
        16:
```
