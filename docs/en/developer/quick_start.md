# Quick Start

This tutorial describes how to quickly develop the Vega algorithm. In this tutorial, a simple CNN network architecture search simple_cnn is used as an example. An operation layer and operation parameters of a small convolutional network are searched by using a random algorithm, and a search data set is Cifar-10.

## 1. Dataset

Before developing an algorithm, you need to determine the data set to which the algorithm is applicable. In this example, the Cifar10 data set is used. You can directly use the Cifar10 data set class provided by Vega.

You need to set data set parameters in the configuration file. Generally, you need to adjust the location of the data set. The data set parameters are as follows:

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

If there is a problem of data loading memory overflow during operation, try to set num_Workers to 0 and set batch_Size to a smaller number.

## 2. Search space

Next, the search space needs to be determined. The search space is related to one or more network definitions, and the content of the search space is a parameter required for constructing the network.

The content parameters of the search space also need to be configured in the configuration file. In this example, the content of the search space is as follows:

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

The definition of the search space in the preceding figure is described as follows:

- type: search space type. The value is fixed to SearchSpace.
- module: list. The custom element indicates a user-defined network.
- custom:
  - type: name of a network class. In this example, the value is SimpleCnnModel, which is the name of a small CNN network model.
  - conv_layer_0:
    - kernel_size: list. Optional range of the convolution kernel
    - bn: list. Indicates whether to add the batch normalization.
    - relu: list. Indicates whether to add the ReLU activation layer.
  - conv_layer_1 and conv_layer_2 are the same as conv_layer_0.
  - fully_connect:
    - output_unit: list. Optional range of the number of output nodes at the fully connected layer.

The SimpleCnnModel network model is defined and implemented in the simple_cnn_model.py file.

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

## 3. Search algorithm

Determine the search algorithm. You can consider the random search algorithm or evolutionary algorithm. In this example, the random algorithm is used.

The search algorithm and parameters must also be configured in the configuration file. Take the simple_cnn random algorithm as an example. The search algorithm parameters are as follows:

```yml
search_algorithm:
    type: RandomSearch
    max_samples: 100
```

This configuration defines the type of the search algorithm (random search).

The search algorithm samples a set of hyperparameters from the search space at a time, and Vega generates an object of the SimpleCnn network model by using the set of hyperparameters.

The random search algorithm is defined and implemented in the random_search.py file. The search interface is used to randomly sample samples in the search space.

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

## 4. Complete code

The complete configuration file is as follows:

```yaml
# my.yml

pipeline: [nas]

nas:
    pipe_step:
        type: NasPipeStep

    dataset:
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

The complete code file is as follows:

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
    vega.run("./my.yaml")
```

## 5. Run the code

Execute the following command:

```bash
python3 ./my.py
```

After running, the tasks directory will be generated in the execution directory. There will be a subdirectory containing time content in this directory. There will be two subdirectories, output and workers under this subdirectory. The output directory will save the network structure description file, and the workers directory will save the network weight file and evaluation results.
