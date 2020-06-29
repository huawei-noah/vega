# Fine Grained Search Space Guide

## 1. Introduction to Fine-Granularity

In most Automl algorithms, the search space is closely related to the network. Each search algorithm defines a series of search spaces and network types that can be identified by the search space. Most of these network types are seldom changed on the basic network, causing that the network cannot be reused. In addition, the search space and the search algorithm are also strongly coupled, and each algorithm has its own definition of the search space. The search space can only be used in a specific scenario, and lacks generality and extension capability.

We analyzed these problems and proposed a fine-grained search space solution.

- The definition of search space can be unified, and the same search space can adapt to different search algorithms.
- The basic network can be reused to provide fine-grained networks. Different types of networks can be constructed in a combined mode.
- The search space can be freely expanded according to the defined network.

## 2. Feature description

### 2.1. Coarse- and fine-grained network definition

We have preset some coarse-grained networks. You can also use some fine-grained blocks to form a coarse-grained network. Algorithm developers can reuse the network based on their requirements without re-implementing the network.

- **ResNet coarse granularit**

```yaml
search_space:
        type: FineGrainedSpace
        modules: ['resnet']
        resnet:
            type: ResNet
            depth: 18  # 18, 34, 50
            num_class: 10
```

- **ResNet fine-grained**

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

### 2.2. Searchable Hyperparameters and Network Structures

In the definition of FineGrainedSpace, any parameter can be used as a hyperparameter for search algorithms.

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

We also search directly for the architectural form of the network.

### 2.3. Different search spaces are used for the same algorithm

We can use the fine-grained searchspace to define the quantification of resnet18 and resnet20.

- **Quantization of resnet20**

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

- **Quantization of resnet18**

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

### 2.4. Reuse of search spaces between different algorithms

In different application scenarios, we hope that the same search space can be used to define the network structure and a few combinations can be used to implement search space reuse.

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

### 2.5. More general search algorithms

Developing. Please wait.

### 2.6. Adaptation of Multiple Backends

Developing. Please wait.

## 3. How to use a fine-grained network

For algorithm developers, we want them to focus on the development of network architecture and hyperparameter search algorithms, rather than on the construction of the network itself. Currently, some FineGrainedSpaces have been preconfigured to provide the hyperparameter and architecture definitions of this type of network. Algorithm developers only need to assemble a new network using the search algorithm based on the description.

Take BacKBoneNas as an example. We hope to reconstruct the Resnet network and perform downsample and doublechannel in some places.

### 3.1 Defining a FineGrainedSpace

To facilitate the invoking of search algorithms, a fine-grained VariantLayer is defined to facilitate the definition of the locations of network downsample and doublechannel.

- Inherit the FineGrainedSpace class.
- Implement the constructor method (which will be directly changed to _init_ in the future). The input parameter is the parameter configuration that can be opened for external search algorithms.
- Multi-layer networks are spliced using the fine-grained Repeat function provided.

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

### 3.2. Using a Configuration File to Form a Model

The input model consists of three parts: init, layer, and head. You can select different blocks/layers/headers for assembling based on the configuration.

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

### 3.3. Define the search algorithm

Backbone_nas is used to search for the positions of doublechannel and downsample in the layer by using the EA algorithm. The parameter space is defined in hyper_parameters of search_space and is processed by the specific search algorithm.

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
