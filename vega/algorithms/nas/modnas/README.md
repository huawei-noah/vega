# ModularNAS

> [MLSys 2021] ModularNAS: Towards Modularized and Reusable Neural Architecture Search

[![Documentation Status](https://readthedocs.org/projects/modularnas/badge/?version=latest)](https://modularnas.readthedocs.io/en/latest/?badge=latest)

[ModularNAS Docs](https://modularnas.readthedocs.io/)

## Introduction

ModularNAS is a neural architecture search (NAS) code library that breaks down state-of-the-art efficient NAS methods into modularized and reusable components, such as search algorithms, evaluation strategies, architecture search space and candidates, and network transformations, while unifies the interactions between components in search procedure.
It also supports automatic search space generation for customized use cases while reusing predefined search strategies, allowing user-friendly NAS deployment with little extra work needed.

<center><img src="docs/images/modnas_formulation.png" style="zoom:30%;" /></center>

Supported NAS methods:

- DARTS [^fn1]
- ProxylessNAS [^fn2]
- Once for All [^fn3]
- Single-Path-One-Shot

Supported search space:

- MobileNetV2 [^fn4] blocks
- MobileNetV3 blocks
- ResNet blocks
- PyramidNet blocks
- ShuffleNetV2 blocks
- Customized network model

## User Guide

### Running ModularNAS on Vega

Vega is a framework for training and fine-tuning deep learning models. For running ModularNAS on Vega, see [this guide](../../../docs/en/algorithms/modnas.md).

### Installation

Requirements:

- Python>=3.5
- numpy
- pyyaml>=5.1

In current directory, run the setup script:

```bash
python3 setup.py install
```

### Running ModularNAS routines

Run an experiment specified by YAML configuration file:

```bash
python3 -m modnas.exec.local -c <path_to_yaml>
```

See documentations for detailed descriptions on config files.

### Search space configuration

The ModularNAS search space can be automatically and procedurally generated from base network model (called backbone), where a chain of constructors specify the behavior of automatic search space generation in sequential steps. We provide various search space components and constructors which can be used to build flexible search spaces on custom models.

Here is a example YAML configuration of the MobileNetV2 search space that defines different expansion ratio and kernel size of the inverted convolution residual convolution in each network block (21 total). The model section defines the base model, and construct section defines a single constructor that generates the search space and the supernet with mixed operators of all candidate operators.

```yaml
model:
  type: CIFAR_MobileNetV2_GPU
mixed_op:
  type: BinaryGateMixedOp
  candidates:
    - MB3E3
    - MB3E6
    - MB5E3
    - MB5E6
    - MB7E3
    - MB7E6
```

Supported base model types are (in model.type):

- ImageNet_MobileNetV2: the original MobileNetV2 architecture on ILSVRC2012 dataset
- CIFAR_MobileNetV2: replace some strided convolution with normal convolution
- ImageNet_MobileNetV2_GPU: MBV2 architecture with increased depth and width
- CIFAR_MobileNetV2_GPU: MBV2 architecture with increased depth and width, and some strides removed

Supported mixed operator types are (in mixed_op.type):

- BinaryGateMixedOp: mixed operator controlled by binary gates, as in ProxylessNAS
- BinaryGateUniformMixedOp: variant of the BinGate where candidates are sampled from uniform distribution in forward pass
- SoftmaxSumMixedOp: returns the sum of the candidate operator outputs weighted by the softmax of some architecture parameter, as in DARTS
- GumbelSumMixedOp: returns the sum of outputs weighted by the gumbel softmax of some parameter, as in SNAS
- IndexMixedOp: returns the output of the candidate selected by the architecture parameter (discrete), as in SPOS, CARS and so on

Supported candidate operator types are (in mixed_op.candidates):

- MBxEy: the inverted mobile residual convolution of kernel size x and expansion ratio of y (e.g., MB3E6)
- NIL: null operation that always returns zero-valued tensor of the input size
- IDT: identity operation that returns the original input

Supported constructor types are (in construct):

- DefaultMixedOpConstructor: convert all placeholder modules in a network to mixed operators (default)
- DefaultOpConstructor: convert all placeholder modules to specified operators
- DefaultSlotArchDescConstructor: convert each placeholder to actual module from list of operator names
- DefaultRecursiveArchDescConstructor: convert each placeholder to actual module in recursive order from list of operator descriptions

### Search algorithm configuration

An architecture search routine requires running an search algorithm to predict the next candidate architecture in a search space, which is implemented as Optimizer component in ModularNAS. We provide implementations of state-of-the-art search algorithms that support efficient architecture search.

Here is an example of Optimizer configuration in YAML that uses ProxylessNAS-G as search algorithm. An optional 'args' section can be provided to configure the optimizer parameters such as learning rate and weight decay.

```yaml
optim:
  type: BinaryGateOptim
  # args:
```

Supported Optimizer types are (in optim.type):

- BinaryGateOptim: ProxylessNAS-G gradient-based search algorithm
- DARTSOptim: DARTS gradient-based search algorithm with second-order approximation
- REINFORCEOptim: REINFORCE policy-gradient search algorithm as in ProxylessNAS-R
- DirectGradOptim: straight-through gradient descent on architecture parameters
- DirectGradOptimBiLevel: straight-through gradient descent on validation dataset
- EvolutionOptim: naïve evolution search with tournament selection
- RegularizedEvolutionOptim: regularized (aging) evolution search algorithm as in [^fn5]
- RandomSearchOptim: random search in a discrete search space
- GridSearchOptim: grid search in a discrete search space

### Evaluation strategy configuration

The candidate architectures sampled by the search algorithm (Optimizer) are evaluated to obtain their performance and efficiency metrics such accuracy, latency and FLOPS. Various evaluation techniques exist that greatly speedup up the process of estimating network performance at the cost of fidelity. We provide implementations of common evaluation strategies in efficient NAS methods as Estimator component in ModularNAS.

Here is an example of Estimator configuration in YAML that specifies a single Estimator component with the name 'search', which treats the network as a supernet and updates the network weights and architecture parameters alternatively in each epoch.

```yaml
estim:
  search:
    type: SuperNetEstim
    epochs: 1
```

Additionally, multiple Estimators can be chained together to carry out complex actions, such as warm starting the network weights before training the architecture parameters.

```yaml
estim:
  warmup:
    type: DefaultEstim
    epochs: 20
  search:
    type: SuperNetEstim
    epochs: 100
```

Supported Estimator types are (in estim.*.type):

- DefaultEstim: train the network weights only
- SuperNetEstim: train the network weights and update architecture parameter alternatively
- SubNetEstim: train and evaluate each candidate architecture separately
- ProgressiveShrinkingEstim: train the network using progressive shrinking (PS) as in Once for All [^fn1]

### Output specifications

The outputs generated by ModularNAS routines are located in "exp/\<exp_name\>/". The output folder may contain the following files:

- logs: logger outputs
- writers: SummaryWriter outputs
- outputs:
  - outputs/arch\_\<name\>\_best.yaml: best architecture description found in Estimator routine named \<name\>

By default, the architecture description is exported by joining the architecture descriptions of each converted module, such as the best candidate of each mixed operator. You can specify alternative export behaviors by using different Exporters. Supported types are (in export.*.type):

- DefaultSlotTraversalExporter: export the architecture description by traversing each placeholder in the network
- DefaultRecursiveExporter: export the arch. desc. by recursively visiting each module in the network
- DefaultParamsExporter: export the key-value pairs of architecture parameters

The construct routine will use search space constructors on the base model in the search phase, and architecture description constructors in the fully training phase.

## Developing

### Registering components

ModularNAS instantiates architecture search components by looking up their names in the registry. Several APIs can be used to register new components under unique names. For example, to register your own model as a new base model for generating search space, simply use the 'register' decorator as follows.

```python
from modnas.registry.arch_space import register

@register
class YourModelClass(nn.Module):
    ...

```

This adds YourModelClass to the registry under the same name. Alternatively, you can use 'register' as a function to specify the name in registry.

```python
register(YourModelClass, 'your_model_name')
```

To instantiate the component in a registry, use the 'build' function as follows:

```python
from modnas.registry.arch_space import build

your_model = build('YourModelClass', *args, **kwargs)
```

Alternatively, use 'get_builder' function to obtain the initializer first:

```python
from modnas.registry.arch_space import get_builder

your_model = get_builder('YourModelClass')(*args, **kwargs)
```

Supported registry paths are:

- arch_space: search space components: base model, mixed operator, primitives, etc.
- construct: Constructor components
- optim: Optimizer components
- estim: Estimator components

### Customizing search space

ModularNAS support automatic search space generation in few lines of code.  To use this feature, simply replace the layer or operator you want to search with a placeholder module named Slot.

Take a convolutional neural network as example. To search for optimal choice for operator rather than convolutions, simply replace the following statement of defining convolution layers:

```python
self.conv = Conv2d(in_channels=C_in, out_channels=C_out, stride=stride)
```

with instantiations of Slot modules:

```python
self.conv = Slot(_chn_in=C_in, _chn_out=C_out, _stride=stride)
```

A search space template is now defined where the convolution operator can be converted to various structures including mixed operator, elastic modules and nested network layers. In the case of using mixed operator, define the search space section in YAML config file as follows:

```yaml
model:
  type: # your model class
construct:
  mixed_op:
    type: DefaultMixedOpConstructor
    args:
      # mixed operator and candidates
```

Now we have a supernet on top of the base model where the original convolution operators are replaced with specified mixed operators and candidates. A search routine can then be set up by matching the search space with selected Optimizer and Estimators.

## Citation

## Reference

[^fn1]: Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. ArXiv, abs/1806.09055, 2019b.

[^fn2]: Cai, H., Zhu, L., and Han, S. Proxylessnas: Direct neural architecture search on target task and hardware. ArXiv, abs/1812.00332, 2019.

[^fn3]: Cai, H., Gan, C., and Han, S. Once for all: Train one network and specialize it for efficient deployment. ArXiv, abs/1908.09791, 2020.

[^fn4]: Sandler, M., Howard, A. G., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4510–4520, 2018.

[^fn5]: Real, E., Aggarwal, A., Huang, Y., and Le, Q. V. Regularized evolution for image classifier architecture search. In AAAI, 2018.
