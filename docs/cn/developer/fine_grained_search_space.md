# 搜索空间和细粒度网络指导

## 1. 细粒度简介

在Automl的大多数算法中搜索空间和网络是强相关的，每种搜索算法都会定义一系列与之识别的搜索空间和网络类型，这些网络类型大都在基础网络上做一些较少的改动，导致网络不能复用。另外，搜索空间和搜索算法也是强耦合的，每个算法都有自己的搜索空间的定义，这种搜索空间只能用于特定的场景，缺乏通用性和扩展能力。

我们对这些问题进行了分析，提出了通用的SearchSpace细粒度网络的方案：

- 能够统一搜索空间的定义方式，同一种搜索空间能够适配不同的搜索算法
- 能够对基础网络进行复用，提供细粒度的网络，通过组合的模式构建出不同形式的网络。
- 搜索空间能够根据定义出来的网络自由扩展。
- 支持多个backend

## 2. 细粒度演示

### 2.1. 使用细粒度构建一个网络

- 继承Module基类，并调用`@ClassFactory.register(ClassType.NETWORK)`注册网络
- 沿用了pytorch的风格，我们会将`self.xx`的变量放入到模块中，默认按照顺序执行。
- 如果需要自定义moduels的执行顺序，可以重写`call`方法

```python
from zeus.common import ClassFactory, ClassType
from zeus.modules.module import Module
from zeus.modules.operators import ops

@ClassFactory.register(ClassType.NETWORK)
class SimpleCnn(Module):
    def __init__(self, block_nums=3, filters=32, kernel_size=3):
        super(SimpleCnn, self).__init__()
        in_channels = 3
        out_channels = filters
        output_size = 32
        for i in range(block_nums):
            block = ConvBlock(in_channels, out_channels, kernel_size)
            self.add_module("block{}".format(i), block)
            in_channels = out_channels
            output_size = (output_size - kernel_size + 1) // 2
        self.fc1 = ops.Linear(in_channels * output_size * output_size, 120)
        self.relu = ops.Relu()
        self.fc2 = ops.Linear(120, 10)

@ClassFactory.register(ClassType.NETWORK)
class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = ops.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = ops.BatchNorm2d(out_channels)
        self.relu = ops.Relu()
        self.pool = ops.MaxPool2d((2, 2))
        
    def call(x):
		x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)

model = SimpleCnn()
print(model)
```

### 2.2.  定义Search Space并使用随机搜索对网络进行搜索

- 利用Vega的pipeline能力

  ```yaml
  pipeline: [hpo]
  
  hpo:
      pipe_step:
          type: SearchPipeStep
  
      search_algorithm:
          type: RandomSearch
  
      search_space:
          type: SearchSpace
          hyperparameters:
              -   key: backbone.block1.conv.in_channels
                  type: CATEGORY
                  range: [8, 16, 32, 64, 128, 256]
      model:
          model_desc:
              modules: ["backbone"]
              backbone:
                  type: SimpleCnn
      dataset:
          type: Cifar10
          common:
              data_path: /cache/datasets/cifar10/
              batch_size: 256
      trainer:
          type: Trainer
          epochs: 1
  ```

- 编写代码单独使用

```python
from vega.algorithms.hpo.random_hpo import RandomSearch
from vega.core.search_space import SearchSpace
from vega.core.search_space.param_types import ParamTypes
from vega.core.search_space.params_factory import ParamsFactory
from zeus.networks.network_desc import NetworkDesc

# SearchSpace的定义
params = ParamsFactory.create_search_space(
    param_name='backbone.block1.conv.in_channels',
    param_type=ParamTypes.CATEGORY,
    param_range=[8, 16, 32, 64, 128, 256])
search_space = SearchSpace().add_hyperparameter(params)
# 搜索算法
id, desc = RandomSearch(search_space).search()
# 解析成模型
model = NetworkDesc(desc).to_model()
print(model)
```

## 3.   网络模块化分组

为了方便网络模块的重用，我们将细粒度的模块按照其功能的不同，进行了分组，每个分组都有其相应的特性。

- **Networks**：定义一个常用的网络，属于粗粒度的网络，如ResNet 和FasterRcnn。网络是其他分组中的子模块。
- **Backbone**：骨干网络。通常采用backbone+ head的模式组成一个网络。在很多场景下我们可以自由的替换不同的backbone已达到处理不同的featureMap。
- **Head**：一般用于特征融合，例如作为分类或者回归问题。这样可以确保更换不同的头，以适应不同的场景。
- **Cells：**组合多个blocks，我们定义了多种Cells来定义组合场景.
- **Blocks**：由基本的算子构成，组合成一个特定功能的block。我们提供给了一些常用的block，这些Block可以用于不同的网络中。
- **Connections**：定义模块之间的连接关系，包括Sequential、Add等，以及一些条件分支的实现语句，如Repeat。 
- **Operators：**定义底层算子，如conv、batch_normal等，我们在此对每个算子做了多个平台的适配，统一了对外的输入输出和接口调用。

例如一个ResNet18的组成如下：

![resnet](../../images/resnet.png)

## 4.  Search Space的定义

Search Space 分为**hyper_parameters**和**condition**两部分：

**hyper_parameters**

用于表示超参的定义，包含key,type和value三个设置：key表示超参的名称，type指定了超参的类型即ParamType，系统根据ParamType选择不同的采样方式。range表示定义的采样的范围。

我们当前预置了如下几种ParamType：

- **INT**： 从一个整数范围上采样一个值，如果range=[0, 10]，表示从0到10中随机采样出一个value

- **INT_EXP：**在整数范围上按照10的指数级采样方式采样一个值，如range=[0, 1000]，会通过log函数映射到[0,10,100,1000]这几个值上

- **INT_CAT**：表示从多个INT类型的数值中选择一个，如range=[16, 32, 64, 128]

- **FLOAT:**  从一个Float范围上采样一个值，如range=[0.001, 1]，采样一个值

- **FLOAT_EXP**：在Float类型范围上按照10的指数级采样方式采样一个值，如range=[0.001, 1]，会通过log函数映射到[1,0.01,0.001]这几个值上

- **FLOAT_CAT :** 表示从多个FLOAT类型的数值中选择一个，如range=[0.1, 0.01, 0.001, 0.99]

- **STRING:** 表示从多个字符串中选择一个，如range=[‘block1’, 'block2', 'block3', 'block4']

  

**condition**

用于表示2个节点之间的关系，当parent满足一定条件时，child节点才会生效

![img](http://hi3ms-image.huawei.com/hi/staticimages/hi3msh/images/2019/0731/15/5d414a699c009.png)![img](http://image.huawei.com/tiny-lts/v1/images/9ed3126327ed5a8abb80_844x290.png@900-0-90-f.png)

这里用一个condition_range来传入条件的值或者范围。具体的：  

- **EQUAL**：condition_range只能包含一个parent的数值，表示child被选择，需要满足parent的值**等于**该数值；
-  **NOT_EQUAL**：condition_range可以包含一个或多个parent的数值，表示child被选择，需要满足parent的值**不等于**condition_range中的提供的所有数值；
- **IN**：如果parent是range类型的，则condition_range必须包含两个值表示该cond_range的最小值和最大值，child被选中必须满足parent当前值落在该cond_range范围内；如果parent是CAT类型的，则condition_range必须包含一个或者多个parent数值，child被选中必须满足parent当前值落在condition_range中的某个数值上。

**forbidden**

用于表示2节点之间的值的互斥关系，节点1含有某个值时，节点2的某些值不会被选择

## 5. 支持多个Backend

我们对底层架构做了封装，统一上层的接口来适配多个不同的backend。其主要核心功能分为：

- **Module**：实现自定义模块的需要继承的基类，统一了各个平台的对于模块内部操作的实现。
- **ops**：上层调用算子的接口，统一了不同平台同一功能算子的命名和输入输出。
- **Serializable：** 对模块中的超参和层次结构进行提取和解析，并序列化成json格式的字典。

![fine_grained_space](../../images/fine_grained_space.png)

## 6. 如何进行细粒度网络的开发

对于算法开发者来说，我们希望其聚焦于网络结构和超参的搜索算法的开发，而不用关心网络本身构建。当前已预置了一些Modules和Networks能够提供该类型网络的超参定义和架构定义的描述，算法开发者只需要根据其描述通过搜索算法装配成新的网络。

### 6.1 定义一个Modules

为了方便大家的使用，我们继承了pytorch的开发习惯，仅仅需要几行的变化就可以成为细粒度中的一个Module。

- 继承Module类，注册到`ClassFactory.register(ClassType.NETWORK)`中
- 使用ops下的算子替换nn下的算子
- 对于顺序执行的网络结构，我们默认会按照self的顺序生成网络，无需再实现forward方法

```python
@ClassFactory.register(ClassType.NETWORK)
class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = ops.conv2d(in_channels, out_channels, kernel_size)
        self.bn = ops.batch_norm2d(out_channels)
        self.relu = ops.relu()
        self.pool = ops.max_pool2d((2, 2))
```

- 如果对于输入需要进行特殊的处理，可以根据自己的需要重写`call`方法

  ```python
  @ClassFactory.register(ClassType.NETWORK)
  class MixedOp(Module):

      def __init__(self, C, stride, ops_cands):
          """Init MixedOp."""
          super(MixedOp, self).__init__()
          self.add_spaces(ops_cands, OPS[ops_cands](C, stride, True))
  
      def call(self, x, weights=None, *args, **kwargs):
          """Call function of MixedOp."""
          if weights is None:
              for model in self.children():
                  x = model(x)
              return x
          return ops.add_n(weights[idx] * op(x) for idx, op in enumerate(self.children()) if weights[idx] != 0)
  ```

### 6.2 使用Connections组装多个模块

我们默认都会采用Sequential的方式组装多个网络，当其他的连接方法时需要手动调用连接的方法。如下面样例采用Add作为两个网络的加和拼接

```python
@ClassFactory.register(ClassType.NETWORK)
class BasicBlock(Module):
    """Create BasicBlock SearchSpace."""

    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1):
        super(BasicBlock, self).__init__()
        base_conv = BasicConv(inchannel,outchannel)
        shortcut = ShortCut(inchannel,outchannel)
        self.add_block = Add(base_conv, shortcut)
        self.relu = ops.relu()
```

开发者也可以自己定义Connections:

- 继承`ConnectionsDecorator`，并注册到`ClassFactory.register(ClassType.NETWORK)`
- init函数接受入参为`*models`，表示接受多个模块，我们会自动调用add_module将这些模块设置到modules中
- 重写`call`方法，通过`self.children()`获取已经添加的模块，并进行详细的操作

```python
@ClassFactory.register(ClassType.NETWORK)
class Sequential(ConnectionsDecorator):
    """Sequential Connections."""

    def __init__(self, *models):
        super(Sequential, self).__init__(*models)

    def compile(self, inputs):
        """Override compile function, conect models into a seq."""
        output = inputs
        models = self.children()
        for model in models:
            output = model(output)
        return output
```
