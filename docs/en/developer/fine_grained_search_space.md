# Search space and Fine-Grained Network guidance
## 1. Fine-grained Introduction
In most Automl algorithms, the search space is closely related to the network. Each search algorithm defines a series of search space and network types that are identified by the search space and network types. Most of these network types are slightly modified on the basic network, resulting in network reuse failure. In addition, the search space and search algorithm are strongly coupled. Each algorithm has its own search space definition. This search space can only be used in specific scenarios and lacks universality and scalability.
After analyzing these problems, we propose a general searchspace fine-grained network solution.

- Unified search space definition mode. The same search space can adapt to different search algorithms.
- Reuses basic networks, provides fine-grained networks, and constructs different types of networks through combinations.
- The search space can be expanded freely based on the defined network.
- Multiple backends are supported.
## 2. Fine-grained demonstration
### 2.1 Building a Network with Fine Grain
- Inherit the Module base class and call `@ClassFactory.register(ClassType.NETWORK)` to register the network.
- The pytorch style is used. The `self.xx` variable is placed in the module. By default, the variable is executed in sequence.
- If you need to customize the execution sequence of modules, rewrite the `call` method.
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
### 2.2. Define Search Space and Use Random Search
- Config in pipeline
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
- Use SearchSpace in code.
```python
from vega.algorithms.hpo.random_hpo import RandomSearch
from vega.core.search_space import SearchSpace
from vega.core.search_space.param_types import ParamTypes
from vega.core.search_space.params_factory import ParamsFactory
from zeus.networks.network_desc import NetworkDesc

# Definition of SearchSpace
params = ParamsFactory.create_search_space(
param_name='backbone.block1.conv.in_channels',
param_type=ParamTypes.CATEGORY,
param_range=[8, 16, 32, 64, 128, 256])
search_space = SearchSpace().add_hyperparameter(params)
# Search algorithm
id, desc = RandomSearch(search_space).search()
# Parse into a model.
model = NetworkDesc(desc).to_model()
print(model)
```
## 3. Module Groups
To facilitate the reuse of network modules, fine-grained modules are grouped based on their functions. Each group has its own features.
- **Networks**: defines a common network, which is a coarse-grained network, such as ResNet and FasterRCNN. Networks are submodules in other groups.
- **Backbone**: backbone network. Generally, the backbone+head mode is used to form a network. In many scenarios, we can flexibly replace different backbones to process different featureMaps.
- **Head**: used for feature fusion, for example, as a classification or regression problem. This ensures that different heads are replaced to accommodate different scenarios.
- **Cells:** Multiple blocks are combined. Multiple cells are defined to define combined scenarios.
- **Blocks**: consists of basic operators and forms a block with specific functions. We provide some common blocks that can be used in different networks.
- **Connections**: defines the connection relationships between modules, including Sequential and Add, and the implementation statements of some condition branches, such as Repeat.
- **Operators:** Defines underlying operators, such as conv and batch_normal. Each operator is adapted to multiple platforms to unify external input, output, and interface invoking.
For example, the composition of a ResNet18 is as follows:
![resnet](../../images/resnet.png)

## 4. Definition of Search Space

The search space consists of **hyper_parameters** and **condition**.
**hyper_parameters**
Specifies the definition of a hyperparameter, including key, type, and value. key indicates the name of a hyperparameter, and type indicates the type of a hyperparameter, that is, ParamType. The system selects a sampling mode based on ParamType. range: specifies the sampling range.
The following param types are preconfigured:

- **INT**: indicates that a value is sampled from an integer range. If the value range is [0, 10], a value is randomly sampled from 0 to 10.
- **INT_EXP:** A value in the integer range is sampled in the exponential sampling mode of 10. For example, if range is [0, 1000], the value is mapped to [0, 10, 100, 1000] through the log function.
- **INT_CAT**: Select a value from multiple INT types, for example, range=[16, 32, 64, 128].
- **FLOAT:** Sampling a value from a floating range. For example, if range is [0.001, 1], a value is sampled.
- **FLOAT_EXP**: sample a value in the Float type range in exponential sampling mode of 10. For example, if range is [0.001, 1], the value is mapped to [1, 0.01, 0.001] through the log function.
- **FLOAT_CAT:** indicates that a value is selected from multiple FLOAT types, for example, range=[0.1, 0.01, 0.001, 0.99].
- **STRING:** indicates that one character string is selected from multiple character strings, for example, range=['block1','block2','block3','block4'].
**condition**
Indicates the relationship between two nodes. A child node takes effect only when the parent node meets certain conditions.
![img](http://hi3ms-image.huawei.com/hi/staticimages/hi3msh/images/2019/0731/15/5d414a699c009.png)![img](http://image.huawei.com/tiny-lts/v1/images/9ed3126327ed5a8abb80_844x290.png@900-0-90-f.png)
The value or range of the condition is transferred by using condition_range. Specifically:
- **EQUAL**: condition_range can contain only one parent value, indicating that the child is selected. The value of parent must be equal to **.
- **NOT_EQUAL**: condition_range can contain one or more values of parent, indicating that child is selected. The value of parent ** must not be equal to all values provided in **condition_range.
- **IN**: If parent is of the range type, condition_range must contain two values, indicating the minimum value and maximum value of cond_range. If child is selected, the current value of parent must be within the range of cond_range. If parent is of the CAT type, condition_range must contain one or more parent values. If child is selected, the current parent value must be within a certain value in condition_range.
**forbidden**
Indicates the mutually exclusive relationship between values of two nodes. If node 1 contains a value, some values of node 2 are not selected.
## 5. Support for Multiple Backends
We encapsulate the underlying architecture and unify upper-layer interfaces to adapt to multiple backends. The core functions are as follows:
- **Module**: base class to be inherited for implementing customized modules, which unifies the implementation of internal module operations on each platform.
- **ops**: upper-layer operator invoking interface, which unifies the names, input, and output of the same functional operator on different platforms.
- **Serializable:** Extracts and parses hyperparameters and hierarchies in the module, and serializes them into a JSON dictionary.
![fine_grained_space](../../images/fine_grained_space.png)

## 6. How to Develop Fine-Grained Networks

For algorithm developers, we want them to focus on the development of search algorithms for network structure and hyperparameters rather than the construction of the network itself. Currently, some modules and networks have been preconfigured that can provide the hyperparameter definition and architecture definition description of this type of network. Algorithm developers only need to assemble new networks using search algorithms based on the description.
### 6.1 Defining a Module
To facilitate your use, we inherit the development habits of pytorch. Only a few lines of changes are required to become a module of fine granularity.
- Inherit the Module class and register it with the `ClassFactory.register(ClassType.NETWORK)`.
- Replace the operator in nn with the operator in ops.
- For the network structure that is executed in sequence, the network is generated in the sequence of self by default, and the forward method does not need to be implemented.
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
- If special processing is required for input, rewrite the `call` method as required.
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
### 6.2 Using Connections to Assemble Multiple Modules
By default, multiple networks are assembled in Sequential mode. When other connection methods are used, you need to manually invoke the connection method. In the following example, Add is used to add and combine the two networks.
```python
@ClassFactory.register(ClassType.NETWORK)
class BasicBlock(Module):
    """Create BasicBlock SearchSpace."""
    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1):
        super(BasicBlock, self).__init__()
        base_conv = BasicConv(inchannel, outchannel)
        shortcut = ShortCut(inchannel, outchannel)
        self.add_block = Add(base_conv, shortcut)
        self.relu = ops.relu()
```
Developers can also define connections as follows:
- Inherit `ConnectionsDecorator` and register with `ClassFactory.register(ClassType.NETWORK)`
- The input parameter of the `init` function is `*models`, indicating that multiple modules are accepted. We will automatically invoke add_module to set these modules to modules.
- Rewrite the `call` method, use `self.children()` to obtain added modules, and perform detailed operations.
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
