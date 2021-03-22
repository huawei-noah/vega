# SEGMENTATION-Adelaide-EA-NAS

## 1. 算法介绍

SEGMENTATION-Adelaide-EA-NAS是图像分割网络架构搜索算法。该搜索算法是在Adelaide沈春华老师研究的Fast-NAS基础上改进的，并将搜索策略从原算法的强化学习(RL)改为遗传算法(EA)搜索，因此命名为SEGMENTATION-Adelaide-EA-NAS。需要了解更多的细节可以看arxiv上的文章 <https://arxiv.org/abs/1810.10804> 。

## 2. 算法原理

现有的语义分割模型一般可以被解耦成编码(encoder)部分（又可称为backbone）和解码（decoder）部分。通常情况下，backbone生成不同尺度的特征图，decoder通过选择不同特征图进行融合并最终上采样至原输入分辨率，进而实现了像素级别的分类即语义分割。SEGMENTATION-Adelaide-EA-NAS主要针对decoder结构进行搜索。

![F3](../../images/Adelaide-EA-NAS1.jpg)

### 2.1 搜索空间和搜索策略

在搜索空间上，主要针对decoder进行搜索，具体包括：（1）decoder和backbone之间的连接；（2）decoder中cell的结构，包括cell中的算子和连接方式；（3）decoder中cell的连接方式。在搜索策略上，采用EA搜索策略。每一个网络由19个字符表示，前13个字符代表的是decoder中cell的结构（包括cell中的算子和连接方式），后6个字符代表的是decoder和backbone之间的连接以及decoder中cell的连接方式。每一代搜索时，改变上一代中的某个字符获得当前代的一个网络表示。此外，考虑在一般情况下对语义分割网络进行训练时，需要对backbone进行预训练（通常是在ImageNet上），然后再在目标分割数据集上对语义分割网络进行训练。因此我们在搜索前，需要对backbone部分进行预训练。这样做的目的是为了加快模型收敛速度，从而加快搜索。下面主要介绍SEGMENTATION-Adelaide-EA-NAS搜索空间：

#### 2.1.1 搜索cell

SEGMENTATION-Adelaide-EA-NAS的搜索空间定义的算子包括：

`conv1x1, conv3x3, sep_conv_3x3, sep_conv_5x5, conv3x3_dil3, conv3x3_dil12, sep_conv_3x3_dil3, sep_conv_5x5_dil6`

如下图所示的两个cell进行融合，融合的方式有两种：一是将两个cell输出的特征图进行concat；二是将两个cell输出的特征图进行“+”。搜索空间中每个cell的channel数是固定的，两特征图进行融合时可能只是分辨率不一致。在搜索空间中，我们定义如果两特征图分辨率不一致，就将低分辨的特征图Upsample至高分辨率特征图的大小，再进行融合即可。

![F3](../../images/Adelaide-EA-NAS3.jpg)

融合的方式和cell的channel数可从配置文件中的agg_size、agg_concat获取。对decoder中cell结构的搜索，包括cell中的算子和连接方式。每一个模型都由19个字符组成，前13个字符代表的是decoder中cell的结构（包括cell中的算子和连接方式）。前13个字符又可以解析成如下的表示方式：

`[op1, [1, 0, op2, op3], [4, 3, op4, op5], [2, 0, op6, op7]]`

，其中op1~op7代表的是算子，如op1为“3”时，就代表了op1此时取的是op_names中的第3个算子（算子编号从0开始）。如配置文件中 `op_names` 定义如下：

`[conv1x1, conv3x3, sep_conv_3x3, sep_conv_5x5, conv3x3_dil3, conv3x3_dil12, sep_conv_3x3_dil3, sep_conv_5x5_dil6]`

，则“3”就代表的是算子`sep_conv_5x5`。除去op1~op7，还有6个数字，代表的是对应算子的input来自哪个输出。比方说，从上面的例子来看，op2对应的是第“1”号输出，op3对应的是第“0”号输出，op4对应的是第“4”号输出，op5对应的是第“3”号输出，op6对应的是第“2”号输出，op7对应的是第“0”号输出。而上述的第0~9号输出就是下图中橙色所标记的算子或符号。

![F3](../../images/Adelaide-EA-NAS4.jpg)

因此，用这13个字符就可以代表decoder cell中的算子和连接方式了。

#### 2.1.2 搜索连接

连接方式搜索包括对decoder和backbone之间的连接以及decoder中cell的连接方式的搜索，可以由6个字符表示。6个字符可以进一步地解析成如下形式：

`[[cell0,cell1], [cell2,cell3], [cell4,cell5]]`

下图是`[[2, 3], [3, 1], [4, 4]]`的可视化形式，`[2,3]`表示的是将backbone输出的第2号特征图和第3号特征图进行融合，融合的方式包括：首先将第2号特征图和第3号特征图分别进入cell进行运算，然后调用MergeCell类的方法将两个特征图进行融合得到第4号特征图；相应地，`[3, 1]` 表示的是将backbone输出的第3号特征图和第1号特征图进行融合，`[4, 4]` 表示的是将上述过程中的第4号特征图和第4号特征图进行融合。最后将所有的融合特征图再进行一次`concat`和`conv1x1`得到网络输出。

![F3](../../images/Adelaide-EA-NAS5.jpg)

特别地，在上述过程中，我们首先将backbone中的所有特征图都通过一个`conv1x1`卷积进行channel的变化，统一成配置文件中的`agg_size`大小。Backbone的特征图编号从0开始。通常我们取Backbone的4层特征图即可。

### 2.2 配置搜索空间

在配置文件中进行参数配置，该文件位于`examples/nas/adelaide_ea/adelaide_ea.yml`：

```yaml
pipeline: [random, mutate]

random:
    search_space:
        type: SearchSpace
        modules: ['custom']
        custom:
            type: AdelaideFastNAS
            backbone_load_path: /cache/models/mobilenet_v2-b0353104.pth
            backbone_out_sizes: [24, 32, 96, 320]
            op_names: [conv1x1, conv3x3, sep_conv_3x3, sep_conv_5x5, conv3x3_dil3, sep_conv_3x3_dil3, sep_conv_5x5_dil6]    # decoder cell中搜索的算子
            agg_size: 64        # decoder中卷积层的channel数
            sep_repeats: 1      # decoder cell中分离式卷积重复的次数
            agg_concat: true    # 如为TRUE，则decoder中，特征融合的方式为concat，否则为“+”
            num_classes: 21
    search_algorithm:
        type: AdelaideRandom
        codec: AdelaideCodec
        max_sample: 100

mutate:
    search_space:
        ref: random.search_space
    search_algorithm:
        type: AdelaideMutate
        codec: AdelaideCodec
        max_sample: 100
```

## 3. 数据要求

适用于图像语义分割数据集，包含RGB图像以及相应的Mask标签。RGB图像值为0至255，Mask标签为0至N-1（N代表的是分割类别数，一般来说255为ignored label，可在配置文件中设置）。相应的数据list为.txt文件，每一行代表了一张图片对应的RGB图像和相应的Mask标签，中间用空格隔开，如：

`VOC2012/JPEGImages/2007_000033.jpg VOC2012/SegmentationClassAug/2007_000033.png`

### 4 算法输出

输出结果包括一系列.pth文件（训练到配置文件中```num_iter```迭代次数的模型）、```result.csv```文件以及```pareto_front.csv```文件。```result.csv```文件记录了所有搜索模型，```pareto_front.csv```文件记录了所有```pareto_front```模型。.csv文件中包含了```encoding```、```flops```、```parameters```以及```mIOU```：

1. ```encoding```：19位字符串表示了模型的结构，19位字符串以“_”为结尾（避免以“0”开头的```encoding```造成记录错误）。

2. ```flops```：记录的是模型的Macc值，如：1371603728表示的就是1.277G。

3. ```parameters```：记录的是模型的parameters值，如：3162900表示的就是3.016M。

4. ```mIOU```：记录的是训练到配置文件中num_iter迭代次数后的模型mIOU。

## 5. Benchmark

请参考 [adelaide_ea.yml](https://github.com/huawei-noah/vega/blob/master/examples/nas/adelaide_ea/adelaide_ea.yml)。
