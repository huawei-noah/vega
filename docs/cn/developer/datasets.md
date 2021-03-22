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

1. 在程序中，使用`ClassFactory`来创建`Dataset`，`mode`来初始化训练集或测试集，通过`Adapter`适配不同框架， 最后通过`loader`来加载数据，如下：

    ```python
    dataset = ClassFactory.get_cls(Classtype.DATASET)
    train_data, test_data = dataset(mode='train'), dataset(mode='test')
    train_data_loader = Adapter(train_data).loader
    test_data_loader = Adapter(test_data).loader
    for input, target in data_loader:
        process_data(input, target)
    ```

### 1.2 架构

Vega的所有数据集类都继承自基类`Dataset`，`Dataset`基类定义了数据集所需的接口， 并提供了`dataloader`、`transforms`、`sampler`等属性，并提供了缺省的实现，派生类可以根据需要来重载这些缺省实现，以下会介绍如何自定义一个 `Dataset`。

## 2. 自定义Dataset

假设用户训练数据集为100张图片，放在10个文件夹中，文件夹名称是分类标签，验证集和测试集也是同样的文件目录。我们需要实现一个名为 `ClassificationDataset` 的数据集类，我们需要按照如下步骤进行:

1. 定义数据集配置。
2. 实现数据集。

## 2.1 定义数据集配置

数据集的配置类为`ClassificationDatasetConfig`，包含四部分：train、val、test、common，在公共配置中有一些缺省的配置项，如下：

```python
from zeus.common import ConfigSerializable


class ClassificationDatasetCommonConfig(ConfigSerializable):
    data_path = None
    batch_size = 1
    shuffle = False
    drop_last = True
    n_class = None
    train_portion = 1.0
    n_images = None
    cached = True
    transforms = []
    num_workers = 1
    distributed = False
    pin_memory = False


class ClassificationDatasetTraineConfig(ClassificationDatasetCommonConfig):
    shuffle = True
    transforms = [
        dict(type='Resize', size=[256, 256]),
        dict(type='RandomCrop', size=[224, 224]),
        dict(type='RandomHorizontalFlip'),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50, 0.5, 0.5], std=[0.50, 0.5, 0.5])]


class ClassificationDatasetValConfig(ClassificationDatasetCommonConfig):
    shuffle = False
    transforms = [
        dict(type='Resize', size=[224, 224]),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50, 0.5, 0.5], std=[0.50, 0.5, 0.5])]


class ClassificationDatasetTestConfig(ClassificationDatasetCommonConfig):
    shuffle = False
    transforms = [
        dict(type='Resize', size=[224, 224]),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50, 0.5, 0.5], std=[0.50, 0.5, 0.5])]


class ClassificationDatasetConfig(ConfigSerializable):
    common = ClassificationDatasetCommonConfig
    train = ClassificationDatasetTraineConfig
    val = ClassificationDatasetValConfig
    test = ClassificationDatasetTestConfig
```

## 2.2 实现Dataset

实现Dataset需要注意：

1. 使用`@ClassFactory.register(ClassType.DATASET)`注册数据类。
2. 重载`__len__()`和`__getitem__()`，提供给dataloader使用。
3. 实现`input_shape()`接口，其返回值要和`__getitem__`的数据的shape相对应。

代码如下：

```python
import numpy as np
import random
import os
import PIL
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.common import FileOps
from zeus.datasets.conf.cls_ds import ClassificationDatasetConfig
from .utils.dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class ClassificationDataset(Dataset):

    config = ClassificationDatasetConfig()

    def __init__(self, **kwargs):
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        sub_path = os.path.abspath(os.path.join(self.args.data_path, self.mode))
        if self.args.train_portion != 1.0 and self.mode == "val":
            sub_path = os.path.abspath(os.path.join(self.args.data_path, "train"))
        if self.args.train_portion == 1.0 and self.mode == "val" and not os.path.exists(sub_path):
            sub_path = os.path.abspath(os.path.join(self.args.data_path, "test"))
        if not os.path.exists(sub_path):
            raise("dataset path is not existed, path={}".format(sub_path))
        self._load_file_indexes(sub_path)
        self._load_data()
        self._shuffle()

    def _load_file_indexes(self, sub_path):
        self.classes = [_file for _file in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, _file))]
        if not self.classes:
            raise("data folder has not sub-folder, path={}".format(sub_path))
        self.n_class = len(self.classes)
        self.classes.sort()
        self.file_indexes = []
        for _cls in self.classes:
            _path = os.path.join(sub_path, _cls)
            self.file_indexes += [(_cls, os.path.join(_path, _file)) for _file in os.listdir(_path)]
        if not self.file_indexes:
            raise("class folder has not image, path={}".format(sub_path))
        self.args.n_images = len(self.file_indexes)
        self.data = None

    def __len__(self):
        return len(self.file_indexes)

    def __getitem__(self, index):
        if self.args.cached:
            (label, _, image) = self.data[index]
        else:
            (label, _file) = self.file_indexes[index]
            image = self._load_image(_file)
        image = self.transforms(image)
        n_label = self.classes.index(label)
        return image, n_label

    def _load_data(self):
        if not self.args.cached:
            return
        self.data = [(_cls, _file, self._load_image(_file)) for (_cls, _file) in self.file_indexes]

    def _load_image(self, image_file):
        img = PIL.Image.open(image_file)
        img = img.convert("RGB")
        return img

    def _to_tensor(self, data):
        if vega.is_torch_backend():
            import torch
            return torch.tensor(data)
        elif vega.is_tf_backend():
            import tensorflow as tf
            return tf.convert_to_tensor(data)

    def _shuffle(self):
        if self.args.cached:
            random.shuffle(self.data)
        else:
            random.shuffle(self.file_indexes)
```

## 2.3 调测

以上实现可以直接用于Vega中的PipeStep，也可以单独调用，单独调用的代码如下：

```python
import unittest
import vega
from zeus.common.class_factory import ClassFactory, ClassType


class TestDataset(unittest.TestCase):

    def test_cifar10(self):
        from zeus.datasets import Adapter
        dataset_cls = ClassFactory.get_cls(ClassType.DATASET, "ClassificationDataset")
        dataset = dataset_cls(mode="train", data_path="/cache/datasets/classification/")
        dataloader = Adapter(dataset).loader
        for input, target in dataloader:
            self.assertEqual(len(input), 1)
            # process(input, target)
            break


if __name__ == "__main__":
    vega.set_backend("pytorch")
    unittest.main()
```

若运行成功，会有如下类似的信息输出：

```text
Ran 1 test in 12.119s

OK
```

## 2.4 完整代码

完整代码可参考：

1. 数据集配置：[cls_ds.py](https://github.com/huawei-noah/vega/blob/master/vega/datasets/conf/cls_ds.py)
2. 数据集实现：[cls_ds.py](https://github.com/huawei-noah/vega/blob/master/vega/datasets/common/cls_ds.py)
