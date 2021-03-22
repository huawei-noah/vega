# 快速开始

本教程提供了如何快速开发Vega算法的教程，以一个简单的`CNN`网络架构搜索为示例来说明，使用随机算法搜索一个小型卷积网络的操作层和操作参数，搜索数据集为Cifar-10。

## 1. 数据集

开发算法，首先要确定该算法适用的数据集，本示例使用的是`Cifar10`数据集，可以直接使用Vega已提供的Cifar10数据集类。

需要在配置文件中配置数据集参数，一般需要调整数据集所在位置，数据集的配置参数如下：

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

如果在运行中出现数据加载内存溢出的问题，请尝试将 num_workers 调整为 0，并将 batch_size 调整为较小的数字。

## 2. 搜索空间

接着需要确定搜索空间，搜索空间和一个或者多个的网络定义相关，搜索空间的内容是构造网络所需要的参数。

搜索空间的内容参数也需要配置在配置文件中，本例的搜索空间内容如下：

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

上图中的搜索空间定义分为两部分，search_space和model，前者描述了超参空间，后者描述基础网络结构。从超参空间中采样，结合网络结构定义，形成一个完整的网络结构。

搜索空间解释如下：

* `blocks`: 表示网络中间叠加多少个`conv+bn+relu`的`block`。
* `channels`: 表示`block`的通道数。

`SimpleCnn`网络模型在`simple_cnn.py`文件中定义和实现。

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

## 3. 搜索算法

可采用随机搜索，

```yml
search_algorithm:
    type: RandomSearch
    policy:
        num_sample: 50
```

RandomSearch是Vega预置的搜索算法。

## 4. 完整的代码

完整的配置文件如下：

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
            params:
                is_grad: False
                sparse: True
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

完整的代码如下：

```python
import vega
import torch.nn as nn
from zeus.common.config import Config
from zeus.common import ClassType, ClassFactory

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

## 5. 运行代码

执行如下命令：

```bash
python3 ./my.py
```

运行结束后，会在执行目录下生成 tasks 目录，在该目录下会有一个包含时间内容的子目录，在该子目录下面有 output 和 workers 两个子目录，其中 output 目录会保存网络结构描述文件，workers 目录会保存该网络的 权重文件 和 评估结果。
