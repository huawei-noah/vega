# 数据集开发指导

## 1. 简介

Vega在`Dataset`类中提供了数据转换和采样相关的接口和公共方法，用户数据处理类可继承自`Dataset`类，使用这些公共能力。

Vega提供了常用的数据集类，包括`Avazu`、`Cifar10`、`Cifar100`、`ImageNet`、`Coco`、`FMnist`、`Mnist`、`Cityscapes`、`Div2K`等，具体描述，可参考[配置参考](../user/config_reference.md)。

### 1.1 使用示例

以下以`Cifar10`为例，来说明如何使用`Dataset`，使用步骤如下：

1. 调整缺省配置，比如要调整数据文件中训练集的位置为本地文件，如下：

    ```yaml
    dataset:
        type: Cifar10
        train:
            data_path: "/cache/datasets/cifar10/"
     ```

1. 在程序中，使用`ClassFactory`来创建`Dataset`，`mode`来初始化训练集或测试集，并使用`Dataloader`来加载数据，如下：

    ```python
    dataset = ClassFactory.get_cls(Classtype.DATASET)
    train_data, test_data = dataset(mode='train'), dataset(mode='test')
    data_loader = train_data.dataloader
    for input, target in data_loader:
        process_data(input, target)
    ```

### 1.2 架构

Vega的所有数据集类都继承自基类`Dataset`，`Dataset`基类定义了数据集所需的接口， 并提供了`dataloader`、`transforms`、`sampler`等属性，并提供了缺省的实现，派生类可以根据需要来重载这些缺省实现，以下会介绍如何自定义一个 `Dataset`。

## 2. 自定义Dataset

假设用户数据为100张图片，放在一个文件夹中，我们需要实现一个名为 `MyDataset` 的数据集类，我们需要按照如下步骤进行:

1. 规划数据集。
2. 实现`Dataloader`。
3. 实现`Transform`。

如上所述，类 `MyDataset` 继承自 `Dataset`，如下：

```python
from vega.datasets.pytorch.common.dataset import Dataset
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.DATASET)
class MyDataset(Dataset):
    def __init__(self， **kwargs):
        super(MyDataset, self).__init__(**kwargs)
```

以上代码中，`@ClassFactory.register(ClassType.DATASET)` 是将 `MyDataset` 注册到`Vega` 库中。

## 2.1 规划数据集

将数据集分为训练集和测试集，训练集用于训练模型，测试集用于验证模型。假设示例中的图片都用于训练，则需要指定一个文件位置的配置参数 `data_path` 。

在模型训练过程中，一般也会动态的将数据集划分为训练集和验证集，需要确定采样方式，顺序采样，还是随机采样，需要增加一个配置参数 `shuffle` 。配置信息如下：

```yaml
    dataset:
        type: MyDataset
        train:
            data_path: "/data/"
            shuffle: false
        valid:
            data_path: "/data/"
            shuffle: false
```

## 2.2 实现Dataloader

假定我们从数据集中每次加载1张图片，每次都从文件加载，使用cv2来加载图片，代码如下：

```python
import cv2


class MyDataset(Dataset):

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_file = self.file[idx]
        img = cv2.imread(img_file)
        return img
```

## 2.3  实现Transform

当前 `Vega` 已提供了多种 `Transform` 供[参考](../user/config_reference.md)。

假设 `MyDataset` 需要实现一个把图片翻转的 `Transform`，输入为一张原始图片，输出为翻转后的图片，假设 `Vega` 并未提供该 `Transform`，我们需要调用 `ImageOps` 的翻转函数来实现，代码如下：

```python
import ImageOps


@TransformFactory.register()
class MyTransform():

    def __call__(self, img):
        return ImageOps.invert(img.convert('RGB'))
```

使用时只需在配置文件中加入该transform即可，如下：

```yaml
dataset:
    type: MyDataset
    train:
        data_path: "/data/dataset/"
        transforms:
            - type: MyTransform
```

若在模型训练过程中调整 `Transfroms` ，可参考[调整Transforms](#transforms2)。

## 2.5 调测

以下是调测新实现的 `MyDataset` 类，代码如下：

```python
import unittest
import torchvision.transforms as  tf
from roma.env import register_roma_env
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType
import vega


@ClassFactory.register(ClassType.PIPE_STEP)
class FakePipeStep(PipeStep, unittest.TestCase):

    def __init__(self):
        PipeStep.__init__(self)
        unittest.TestCase.__init__(self)

    def do(self):
        dataset = ClassFactory.get_cls(ClassType.DATASET)(mode="train")
        train = dataset.dataloader
        self.assertEqual(len(train), 100)
        for input, target in train:
            self.assertEqual(len(input), 1)
            break


class TestDataset(unittest.TestCase):

    def test_cifar10(self):
        vega.run('./dataset.yml')


if __name__ == "__main__":
    unittest.main()
```

若运行成功，会有如下类似的信息输出：

```text
Ran 1 test in 12.119s

OK
```

## 2.6 完整代码

配置文件：

```yaml
pipeline: [fake]

fake:
    pipe_step:
        type: FakePipeStep

    dataset:
        type: MyDataset
        train:
            data_path: "/data/dataset/train/"
            shuffle: false
            transform:
                - type: MyTransform
        valid:
            data_path: "/data/dataset/valid/"
            shuffle: false
```

代码：

```python

import cv2


class MyDataset(Dataset):

    def __init__(self, **kwargs):
    """Construct the MyDataset class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)

    def __len__(self):
    """Get the length of the dataset."""
        return len(self.file)

    def __getitem__(self, idx):
    """Get an item of the dataset according to the index."""
        img_file = self.file[idx]
        img = cv2.imread(img_file)
        return img

```

## 3. 参考

<span id=transform2></span>

1. 初始化 `dataset` 时指定Transforms

   ```python
    dataset = ClassFactory.get_cls(ClassType.DATASET)(
        mode="train",
        transforms=[tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip()]
        )
   ```

1. 在模型训练过程中动态调整 `Transforms`

   提供了 `append()`， `insert()`, `remove()`, `replace()` 等方法，分别提供了追加、插入、删除和替换方法，如下:

    ```python
    dataset.transforms.append(tf.ToTensor())
    dataset.transforms.insert(2, "Color", level=2)
    dataset.transforms.remove("Color")
    dataset.transforms.replace(
        [tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip()]
        )
    ```
