# 快速开始

本教程提供了如何快速开发Vega算法的教程，以一个简单的`CNN`网络架构搜索`simple_cnn`为示例来说明，使用随机算法搜索一个小型卷积网络的操作层和操作参数，搜索数据集为Cifar-10。

## 1. 数据集

开发算法，首先要确定该算法适用的数据集，本示例使用的是`Cifar10`数据集，可以直接使用Vega已提供的Cifar10数据集类。

需要在配置文件中配置数据集参数，一般需要调整数据集所在位置，数据集的配置参数如下：

```yaml
dataset:
    type: Cifar10
    common:
        data_path: '/dataset/cifar10/'
    train:
        shuffle: False
        num_workers: 8
        batch_size: 256
        train_portion: 0.9
    valid:
        shuffle: False
        num_workers: 8
        batch_size: 256
        train_portion: 0.9
```

如果在运行中出现数据加载内存溢出的问题，请尝试将 num_workers 调整为 0，并将 batch_size 调整为较小的数字。

## 2. 搜索空间

接着需要确定搜索空间，搜索空间和一个或者多个的网络定义相关，搜索空间的内容是构造网络所需要的参数。

搜索空间的内容参数也需要配置在配置文件中，本例的搜索空间内容如下：

```yaml
search_space:
    type: SearchSpace
    module: ['custom']
    custom:
        type: SimpleCnnModel
        conv_layer_0:
            kernel_size: [1, 3, 5]
            bn: [False, True]
            relu: [False, True]
        conv_layer_1:
            kernel_size: [1, 3, 5]
            bn: [False, True]
            relu: [False, True]
        conv_layer_2:
            kernel_size: [1, 3, 5]
            bn: [False, True]
            relu: [False, True]
        fully_connect:
            output_unit: [16, 20, 24, 28, 32]
```

上图中的搜索空间定义解释如下：

* `type`：搜索空间类型，固定为`SearchSpace`。
* `module`：列表。里面的元素`custom`表示这是一个自定义的网络。
* `custom`:
  * `type`：网络类的名称，此处为`SimpleCnnModel`，它是一个小型CNN网络模型的类名。
  * `conv_layer_0`:
    * `kernel_size`: 列表。表示卷积核的可选范围
    * `bn`: 列表。表示是否要加入`batch normalization`
    * `relu`: 列表。表示是否要加入`ReLU`激活层
  * `conv_layer_1`和`conv_layer_2`同`conv_layer_0`。
  * `fully_connect`:
    * `output_unit`: 列表。表示全连接层的输出节点数的可选范围。

`SimpleCnnModel`网络模型在`simple_cnn_model.py`文件中定义和实现。

```python
@NetworkFactory.register(NetTypes.CUSTOM)
class SimpleCnn(nn.Module):

    def __init__(self, desc):
        super(SimpleCnn, self).__init__()
        self.conv_num = 3
        self.conv_layers = nn.ModuleList([None] * self.conv_num)
        self.bn_layers = nn.ModuleList([None] * self.conv_num)
        self.relu_layers = nn.ModuleList([None] * self.conv_num)
        self.pool_layers = nn.ModuleList([None] * self.conv_num)
        conv_layer_names = ["conv_layer_{}".format(i) for i in range(self.conv_num)]
        inp_filters = 3
        out_size = 32
        for i, key in enumerate(conv_layer_names):
            out_filters = desc[key]['filters']
            kernel_size = desc[key]['kernel_size']
            padding = (kernel_size - 1) // 2
            self.conv_layers[i] = nn.Conv2d(inp_filters, out_filters, padding=padding, kernel_size=kernel_size)
            if 'bn' in desc[key].keys():
                if desc[key]['bn']:
                    self.bn_layers[i] = nn.BatchNorm2d(out_filters)
            if 'relu' in desc[key].keys():
                if desc[key]['relu']:
                    self.relu_layers[i] = nn.ReLU(inplace=False)
            self.pool_layers[i] = nn.MaxPool2d(2, stride=2)
            inp_filters = out_filters
            out_size = out_size // 2
        fc_inp_size = inp_filters * out_size * out_size
        fc_out_size = desc['fully_connect']['output_unit']
        self.fc0 = nn.Linear(fc_inp_size, fc_out_size)
        self.fc0_relu = nn.ReLU(inplace=True)
        fc_inp_size = fc_out_size
        fc_out_size = 10
        self.fc1 = nn.Linear(fc_inp_size, fc_out_size)

    def forward(self, x):
        for i in range(self.conv_num):
            x = self.conv_layers[i](x)
            if self.bn_layers[i] is not None:
                x = self.bn_layers[i](x)
            if self.relu_layers[i] is not None:
                x = self.relu_layers[i](x)
            x = self.pool_layers[i](x)
        x = self.fc0(x.view(x.size(0), -1))
        x = self.fc0_relu(x)
        x = self.fc1(x)
        return x
```

## 3. 搜索算法

确定搜索算法，可考虑随机搜索，或者进化算法，本例我们使用随机算法。

搜索算法的选择和参数同样也要配置在配置文件中，以simple_cnn的随机算法为例，它的搜索算法参数内容如下：

```yml
search_algorithm:
    type: RandomSearch
    max_samples: 100
```

该配置内容定义了搜索算法的类型（随机搜索）.

搜索算法每次从搜索空间中采样一组超参数，Vega通过这组超参数生成一个SimpleCnn网络模型的对象。

随机搜索算法在`random_search.py`文件中定义和实现，其中`search`接口负责每一次随机采样搜索空间里的样本。

```python
@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class RandomSearch(SearchAlgorithm):
    def __init__(self, search_space):
        super(RandomSearch, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space.search_space)
        self.max_samples = self.cfg["max_samples"]
        self.sample_count = 0

    def _sub_config(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self._sub_config(value)
            elif isinstance(value, list):
                choice = random.choice(value)
                config[key] = choice
        return config

    def search(self):
        desc = {}
        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            module_cfg = self._sub_config(config_space)
            desc[key] = module_cfg
        desc = update_dict(desc, self.search_space)
        self.sample_count += 1
        self._save_model_desc_file(self.sample_count, desc)
        return self.sample_count, NetworkDesc(desc)
```

## 4. 完整的代码

完整的配置文件如下：

```yaml
# my.yml

pipeline: [nas]

nas:
    pipe_step:
        type: NasPipeStep

    dataset:
        type: Cifar10
        common:
            data_path: '/cache/datasets/cifar10/'
        train:
            shuffle: False
            num_workers: 8
            batch_size: 256
            train_portion: 0.9
        valid:
            shuffle: False
            num_workers: 8
            batch_size: 256
            train_portion: 0.9

    search_space:
        type: SearchSpace
        modules: ['custom']
        custom:
            name: SimpleCnn
            conv_layer_0:
                kernel_size: [1, 3, 5]
                bn: [False, True]
                relu: [False, True]
                filters: [8, 16, 32]
            conv_layer_1:
                kernel_size: [1, 3, 5]
                bn: [False, True]
                relu: [False, True]
                filters: [8, 16, 32]
            conv_layer_2:
                kernel_size: [1, 3, 5]
                bn: [False, True]
                relu: [False, True]
                filters: [8, 16, 32]
            fully_connect:
                output_unit: [16, 20, 24, 28, 32]

    search_algorithm:
        type: RandomSearch
        max_samples: 20

    trainer:
        type: Trainer
        epochs: 20
```

完整的代码如下：

```python
# my.py

import os
import copy
import json
import random
import torch.nn as nn
import vega
from vega.core.common.utils import update_dict
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import UserConfig, TaskOps, FileOps
from vega.search_space.networks import NetTypes, NetworkFactory, NetworkDesc
from vega.search_space.search_algs import SearchAlgorithm


@NetworkFactory.register(NetTypes.CUSTOM)
class SimpleCnn(nn.Module):

    def __init__(self, desc):
        super(SimpleCnn, self).__init__()
        self.conv_num = 3
        self.conv_layers = nn.ModuleList([None] * self.conv_num)
        self.bn_layers = nn.ModuleList([None] * self.conv_num)
        self.relu_layers = nn.ModuleList([None] * self.conv_num)
        self.pool_layers = nn.ModuleList([None] * self.conv_num)
        conv_layer_names = ["conv_layer_{}".format(i) for i in range(self.conv_num)]
        inp_filters = 3
        out_size = 32
        for i, key in enumerate(conv_layer_names):
            out_filters = desc[key]['filters']
            kernel_size = desc[key]['kernel_size']
            padding = (kernel_size - 1) // 2
            self.conv_layers[i] = nn.Conv2d(inp_filters, out_filters, padding=padding, kernel_size=kernel_size)
            if 'bn' in desc[key].keys():
                if desc[key]['bn']:
                    self.bn_layers[i] = nn.BatchNorm2d(out_filters)
            if 'relu' in desc[key].keys():
                if desc[key]['relu']:
                    self.relu_layers[i] = nn.ReLU(inplace=False)
            self.pool_layers[i] = nn.MaxPool2d(2, stride=2)
            inp_filters = out_filters
            out_size = out_size // 2
        fc_inp_size = inp_filters * out_size * out_size
        fc_out_size = desc['fully_connect']['output_unit']
        self.fc0 = nn.Linear(fc_inp_size, fc_out_size)
        self.fc0_relu = nn.ReLU(inplace=True)
        fc_inp_size = fc_out_size
        fc_out_size = 10
        self.fc1 = nn.Linear(fc_inp_size, fc_out_size)

    def forward(self, x):
        for i in range(self.conv_num):
            x = self.conv_layers[i](x)
            if self.bn_layers[i] is not None:
                x = self.bn_layers[i](x)
            if self.relu_layers[i] is not None:
                x = self.relu_layers[i](x)
            x = self.pool_layers[i](x)
        x = self.fc0(x.view(x.size(0), -1))
        x = self.fc0_relu(x)
        x = self.fc1(x)
        return x


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class RandomSearch(SearchAlgorithm):
    def __init__(self, search_space):
        super(RandomSearch, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space.search_space)
        self.max_samples = self.cfg["max_samples"]
        self.sample_count = 0

    def _sub_config(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self._sub_config(value)
            elif isinstance(value, list):
                choice = random.choice(value)
                config[key] = choice
        return config

    def search(self):
        desc = {}
        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            module_cfg = self._sub_config(config_space)
            desc[key] = module_cfg
        desc = update_dict(desc, self.search_space)
        self.sample_count += 1
        self._save_model_desc_file(self.sample_count, desc)
        return self.sample_count, NetworkDesc(desc)

    def update(self, worker_path):
        """Update SimpleRand."""
        pass

    @property
    def is_completed(self):
        """Check if the search is finished."""
        return self.sample_count >= self.max_samples

    def _save_model_desc_file(self, id, desc):
        output_path = TaskOps(UserConfig().data.general).local_output_path
        desc_file = os.path.join(output_path, "nas", "model_desc_{}.json".format(id))
        FileOps.make_base_dir(desc_file)
        output = {}
        for key in desc:
            if key in ["type", "modules", "custom"]:
                output[key] = desc[key]
        with open(desc_file, "w") as f:
            json.dump(output, f)


if __name__ == "__main__":
    vega.run("./my.yml")
```

## 5. 运行代码

执行如下命令：

```bash
python3 ./my.py
```

运行结束后，会在执行目录下生成 tasks 目录，在该目录下会有一个包含时间内容的子目录，在该子目录下面有 output 和 workers 两个子目录，其中 output 目录会保存网络结构描述文件，workers 目录会保存该网络的 权重文件 和 评估结果。
