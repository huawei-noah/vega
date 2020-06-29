# 细粒度搜索空间指导

## 1. 细粒度简介

在Automl的大多数算法中搜索空间和网络是强相关的，每种搜索算法都会定义一系列与之识别的搜索空间和网络类型，这些网络类型大都在基础网络上做一些较少的改动，导致网络不能复用。另外，搜索空间和搜索算法也是强耦合的，每个算法都有自己的搜索空间的定义，这种搜索空间只能用于特定的场景，缺乏通用性和扩展能力。

我们对这些问题进行了分析，提出了细粒度的Search Space方案：

- 能够统一搜索空间的定义方式，同一种搜索空间能够适配不同的搜索算法
- 能够对基础网络进行复用，提供细粒度的网络，通过组合的模式构建出不同形式的网络。
- 搜索空间能够根据定义出来的网络自由扩展。

## 2. 特征说明

### 2.1.   自由选择粗粒度和细粒度来定义网络

我们预置了一些粗粒度的网络，也可以通过一些细粒度的block去组成粗粒度的网络。算法开发人员可以根据自己的需要去复用网络，而不需要重新实现。

- **resnet粗粒度表示：**

```yaml
search_space:
        type: FineGrainedSpace
        modules: ['resnet']
        resnet:
            type: ResNet
            depth: 18 #18, 34, 50  确定网络
            num_class: 10
```

- **resnet细粒度表示：**

```yaml
search_space:
        type: FineGrainedSpace
        modules: ['init', 'bottlelayer', 'pooling', 'view', 'linear']
        init:
            type: InitialBlock
            init_plane: 64
        bottlelayer:
            type: BasicLayer
            in_plane: 64
            expansion: 4
            block: 
                type: BottleneckBlock
            layer_reps: [3, 4, 6, 3]
        pooling:
            type: AdaptiveAvgPool2d
            output_size: !!python/tuple [1, 1]
        view:
            type: View
        linear:
            type: Linear
            in_features: 2048
            out_features: 10
```

### 2.2. 可供搜索的超参和网络结构

在FineGrainedSpace的定义中，任何的参数都可作为超参供搜索算法使用。

```yaml
search_space:
        type: FineGrainedSpace

        hyper_parameters:
            resnet18.doublechannel: [0, 0, 1, 0, 1, 0, 1, 0]
            resnet18.downsample: [0, 0, 1, 0, 1, 0, 1, 0]

        modules: ['resnet18']
        resnet18:
            type: ResNetVariant
            base_depth: 18
            base_channel: 64
            doublechannel: [0, 0, 1, 0, 1, 0, 1, 0]
            downsample: [0, 0, 1, 0, 1, 0, 1, 0]
```

我们也直接搜索网络的架构形式。

### 2.3. 同一个算法使用不同的search space

我们可以使用细粒度的searchspace来定义对resnet18和resnet20两种不同网络进行量化处理。

- **resnet20的量化**

```yaml
search_space:
        type: FineGrainedSpace

        modules: ['resnet', 'process']
        resnet:
            type: ResNet
            depth: 20
            block:
                type: BasicBlock
            init_plane: 16
            out_plane: 64
            num_reps: 9
            items:
                inchannel: [16, 16, 16, 16, 32, 32, 32, 64, 64]
                outchannel: [16, 16, 16, 32, 32, 32, 64, 64, 64]
                stride: [1, 1, 1, 2, 1, 1, 2, 1, 1]
        process:
            type: Quant
            nbit_w_list: [8, 4, 8, 4, 4, 8, 8, 8, 8, 4, 4, 8, 4, 4, 4, 4, 8, 4, 8, 8]
            nbit_a_list: [8, 4, 4, 8, 8, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 4, 8, 4, 8, 8]
```

- **resnet18的量化**

```yaml
 search_space:
        type: FineGrainedSpace

        modules: ['resnet', 'process']
        resnet:
            type: ResNet
            depth: 18
            block:
                type: BasicBlock
        process:
            type: Quant
            nbit_w_list: [8, 4, 8, 4, 4, 8, 8, 8, 8, 4, 4, 8, 4, 4, 4, 4, 8, 4, 8, 8]
            nbit_a_list: [8, 4, 4, 8, 8, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 4, 8, 4, 8, 8]
```



### 2.4.  不同的算法之间的search space复用

在不同的应用场景中，我们希望能够使用同样的searchspace来定义网络结构，通过少量的组合来实现searchspace的复用。

- **quant**

```yaml
search_space:
        type: FineGrainedSpace

        modules: ['resnet', 'process']
        resnet:
            type: ResNet
            depth: 20
            block:
                type: BasicBlock
            init_plane: 16
            out_plane: 64
            num_reps: 9
            items:
                inchannel: [16, 16, 16, 16, 32, 32, 32, 64, 64]
                outchannel: [16, 16, 16, 32, 32, 32, 64, 64, 64]
                stride: [1, 1, 1, 2, 1, 1, 2, 1, 1]
        process:
            type: Quant
            nbit_w_list: [8, 4, 8, 4, 4, 8, 8, 8, 8, 4, 4, 8, 4, 4, 4, 4, 8, 4, 8, 8]
            nbit_a_list: [8, 4, 4, 8, 8, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 4, 8, 4, 8, 8]
```

- **prune**

```yaml
search_space:
        type: FineGrainedSpace

        modules: ['resnet', 'process']
        resnet:
          type: ResNet
          depth: 20
          block:
            type: BasicBlock
          init_plane: 16
          out_plane: 60
          num_reps: 9
          items:
            inchannel: [16, 15, 15, 15, 29, 29, 29, 60, 60]
            outchannel: [15, 15, 15, 29, 29, 29, 60, 60, 60]
            inner_plane: [14, 16, 16, 29, 30, 30, 62, 60, 62]
            stride: [1, 1, 1, 2, 1, 1, 2, 1, 1]
        process:
          type: Prune
          chn_node_mask: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
          chn_mask: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
          init_model_file: ./bestmodel.pth
```

### 2.5.  更加通用的搜索算法

正在开发中，敬请等待。

### 2.6. 多Backend的适配

正在开发中，敬请等待。

## 3. 如何使用一个细粒度网络

对于算法开发者来说，我们希望其聚焦于网络结构和超参的搜索算法的开发，而不用关心网络本身构建。当前已预置了一些FineGrainedSpace能够提供该类型网络的超参定义和架构定义的描述，算法开发者只需要根据其描述通过搜索算法装配成新的网络。

以BacKBoneNas为例，我们希望对Resnet网络进行改造，在某些位置进行downsample和doublechannel

### 3.1 定义一个FineGrainedSpace

为了方便搜索算法的调用，我们定义了一个细粒度的VariantLayer方便定义网络downsample和doublechannel的位置。

- 继承FineGrainedSpace类
- 实现constructor方法（后续直接改用_init_)，其入参为可对外搜索算法开放的参数配置。
- 使用细粒度的提供的Repeat函数拼接多层网络

```python
@ClassFactory.register(ClassType.SEARCH_SPACE)
class VariantLayer(FineGrainedSpace):
    """Create VariantLayer SearchSpace."""

    def constructor(self, in_plane, out_plane, doublechannel, downsample, expansion, block, groups=1, base_width=64):
        items = {}
        inchannel = []
        outchannel = []
        stride = []
        num_reps = len(doublechannel)
        for i in range(num_reps):
            inchannel.append(in_plane)
            out_plane = out_plane if doublechannel[i] == 0 else out_plane * 2
            outchannel.append(out_plane)
            in_plane = out_plane * expansion
            if downsample[i] == 0:
                stride.append(1)
            else:
                stride.append(2)
        items['inchannel'] = inchannel
        items['outchannel'] = outchannel
        items['stride'] = stride
        items['groups'] = [groups] * num_reps
        items['base_width'] = [base_width] * num_reps
        self.repeat = Repeat(num_reps=num_reps, items=items, ref=block)
```

### 3.2. 使用配置文件组成模型

我们需要输入的模型由三部分组成，init、layer、和head。可通过配置根据具体的要求选择不同的block/layer/head进行拼装。

```yaml
search_space:
        type: FineGrainedSpace
        modules: ['init', 'layer', 'head']
        init:
            type: SmallInputInitialBlock
            init_plane: 32
        layer:
            type: VariantLayer
            block:
                type: BasicBlock
            in_plane: 32
            out_plane: 512
            doublechannel: [0, 1, 1, 0, 1, 0, 0, 1]
            downsample : [1, 1, 0, 1, 0, 0, 0, 1]
            expansion: 1
        head:
            type: LinearClassificationHead
            base_channel: 512
            num_class: 10
```

### 3.3. 定义搜索算法

backbone_nas是通过ea算法对layer中的doublechannel和downsample的位置进行搜索，在search_space的hyper_parameters中定义参数的空间，由具体的搜索算法处理。

```python
search_algorithm:
        type: BackboneNasSearch

search_space:
        type: FineGrainedSpace

        hyper_parameters:
            layer.doublechannel: [0, 1, 1, 0, 1, 0, 0, 1]
            layer.downsample : [1, 1, 0, 1, 0, 0, 0, 1]

        modules: ['init', 'layer', 'head']
        init:
            type: SmallInputInitialBlock
            init_plane: 32
        layer:
            type: VariantLayer
            block:
                type: BasicBlock
            in_plane: 32
            out_plane: 512
            doublechannel: [0, 1, 1, 0, 1, 0, 0, 1]
            downsample : [1, 1, 0, 1, 0, 0, 0, 1]
            expansion: 1
        head:
            type: LinearClassificationHead
            base_channel: 512
            num_class: 10
```
