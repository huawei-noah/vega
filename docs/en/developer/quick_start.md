# Quick Start

This tutorial describes how to quickly develop the Vega algorithm. In this tutorial, a simple CNN network architecture search is used as an example. An operation layer and operation parameters of a small convolutional network are searched by using a random algorithm, and a search data set is Cifar-10.

## 1. Dataset

Before developing an algorithm, you need to determine the data set to which the algorithm is applicable. In this example, the Cifar10 data set is used. You can directly use the Cifar10 data set class provided by Vega.

You need to set data set parameters in the configuration file. Generally, you need to adjust the location of the data set. The data set parameters are as follows:

```yaml
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
```

If there is a problem of data loading memory overflow during operation, try to set num_Workers to 0 and set batch_Size to a smaller number.

## 2. Search space

Next, the search space needs to be determined. The search space is related to one or more network definitions, and the content of the search space is a parameter required for constructing the network.

The content parameters of the search space also need to be configured in the configuration file. In this example, the content of the search space is as follows:

```yaml
search_space:
    hyperparameters:
        -   key: network.backbone.blocks
            type: CATEGORY
            range: [1, 2, 3, 4]
        -   key: network.backbone.channels
            type: CATEGORY
            range:  [32, 48, 56, 64]

model:
    model_desc:
        modules: [backbone]
        backbone:
            type: SimpleCnn
            num_class: 10
            fp16: False
```

The search space definition in the preceding figure is divided into two parts: search_space and model. The former describes the hyperparameter space, and the latter describes the basic network structure. The data is sampled from the hyperparameter space and combined with the network structure definition to form a complete network structure.

The search space is described as follows:

- blocks: Indicates the number range of blocks of conv+bn+relu.
- channels: Indicates the range of block's channel.

The SimpleCnn network model is defined and implemented in the simple_cnn.py file.

```python
@ClassFactory.register(ClassType.NETWORK)
class SimpleCnn(nn.Module):
    """Simple CNN network."""

    def __init__(self, **desc):
        """Initialize."""
        super(SimpleCnn, self).__init__()
        desc = Config(**desc)
        self.num_class = desc.num_class
        self.fp16 = desc.get('fp16', False)
        self.blocks = desc.blocks
        self.channels = desc.channels
        self.conv1 = nn.Conv2d(3, 32, padding=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.blocks = self._blocks(self.channels)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(self.channels, 64, padding=1, kernel_size=3)
        self.global_conv = nn.Conv2d(64, 64, kernel_size=8)
        self.fc = nn.Linear(64, self.num_class)

    def _blocks(self, out_channels):
        blocks = nn.ModuleList([None] * self.blocks)
        in_channels = 32
        for i in range(self.blocks):
            blocks[i] = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            in_channels = out_channels
        return blocks

    def forward(self, x):
        """Forward."""
        x = self.pool1(self.conv1(x))
        for block in self.blocks:
            x = block(x)
        x = self.global_conv(self.conv2(self.pool2(x)))
        x = self.fc(x.view(x.size(0), -1))
        return x
```

## 3. Search algorithm

You can use the random search mode. The configuration is as follows:

```yml
search_algorithm:
    type: RandomSearch
    policy:
        num_sample: 50
```

RandomSearch is a preset search algorithm of Vega.

## 4. Complete code

The complete configuration file is as follows:

```yaml
# my.yml

pipeline: [nas]

nas:
    pipe_step:
        type: SearchPipeStep
    search_algorithm:
        type: RandomSearch
        policy:
            num_sample: 50

    search_space:
        hyperparameters:
            -   key: network.backbone.blocks
                type: CATEGORY
                range: [1, 2, 3, 4]
            -   key: network.backbone.channels
                type: CATEGORY
                range:  [32, 48, 56, 64]

    model:
        model_desc:
            modules: [backbone]
            backbone:
                type: SimpleCnn
                num_class: 10
                fp16: False

    trainer:
        type: Trainer
        optimizer:
            type: SGD
            params:
                lr: 0.01
                momentum: 0.9
        lr_scheduler:
            type: MultiStepLR
            params:
                warmup: False
                milestones: [30]
                gamma: 0.5
        loss:
            type: CrossEntropyLoss

        metric:
            type: accuracy
        epochs: 3
        save_steps: 250
        distributed: False
        num_class: 10
    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
            batch_size: 64
            num_parallel_batches: 64
            fp16: False
```

The complete code file is as follows:

```python
import vega
import torch.nn as nn
from vega.common.config import Config
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class SimpleCnn(nn.Module):
    """Simple CNN network."""

    def __init__(self, **desc):
        """Initialize."""
        super(SimpleCnn, self).__init__()
        desc = Config(**desc)
        self.num_class = desc.num_class
        self.fp16 = desc.get('fp16', False)
        self.blocks = desc.blocks
        self.channels = desc.channels
        self.conv1 = nn.Conv2d(3, 32, padding=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.blocks = self._blocks(self.channels)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(self.channels, 64, padding=1, kernel_size=3)
        self.global_conv = nn.Conv2d(64, 64, kernel_size=8)
        self.fc = nn.Linear(64, self.num_class)

    def _blocks(self, out_channels):
        blocks = nn.ModuleList([None] * self.blocks)
        in_channels = 32
        for i in range(self.blocks):
            blocks[i] = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            in_channels = out_channels
        return blocks

    def forward(self, x):
        """Forward."""
        x = self.pool1(self.conv1(x))
        for block in self.blocks:
            x = block(x)
        x = self.global_conv(self.conv2(self.pool2(x)))
        x = self.fc(x.view(x.size(0), -1))
        return x


if __name__ == "__main__":
    vega.run("./my.yml")
```

## 5. Run the code

Execute the following command:

```bash
python3 ./my.py
```

After running, the tasks directory will be generated in the execution directory. There will be a subdirectory containing time content in this directory. There will be two subdirectories, output and workers under this subdirectory. The output directory will save the network structure description file, and the workers directory will save the network weight file and evaluation results.
