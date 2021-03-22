# Dataset Development Guide

## 1. Introduction

The Vega provides interfaces and public methods related to data conversion and sampling in the Dataset class. The user data processing class can inherit from the Dataset class and use these public capabilities.

Vega provides common dataset classes, including Avazu, Cifar10, Cifar100, ImageNet, Coco, FMnist, Mnist, Cityscapes, and Div2K. For details, see the  HYPERLINK "../user/config_reference.md"  Configuration Reference.Error! Hyperlink reference not valid.

### 1.1 Example

The following describes how to use the Dataset by using the cifar10 as an example. The procedure is as follows:

1. Adjust the default configuration. For example, change the location of the training set in the data file to a local file.

    ```yaml
    dataset:
        type: Cifar10
        train:
            data_path: "/cache/datasets/cifar10/"
     ```

2. In the program, use ClassFactory to create a Dataset, mode to initialize the training set or test set, use Adapter to adapte to difference backend, and use loader to get data, as shown in the following figure.

    ```python
    dataset = ClassFactory.get_cls(Classtype.DATASET)
    train_data, test_data = dataset(mode='train'), dataset(mode='test')
    train_data_loader = Adapter(train_data).loader
    test_data_loader = Adapter(test_data).loader
    for input, target in data_loader:
        process_data(input, target)
    ```

### 1.2 Architecture

All dataset classes of Vega are inherited from the base class Dataset. The base class Dataset defines the interfaces required by the dataset, provides attributes such as dataloader, transforms, and sampler, and provides default implementation. The derived class can reload these default implementations as required. The following describes how to customize a dataset.

## 2. Customize a dataset

Assume that the user training dataset contains 100 images and is stored in 10 folders. The folder name is the category label. The verification set and test set are the same file directory. To implement a dataset class named ClassificationDataset, perform the following steps:

1. Define dataset configurations.
2. Implement dataset class.

## 2.1 Defining Dataset Configurations

The configuration class of the dataset is ClassificationDatasetConfig, which consists of four parts: train, val, test, and common. There are some default configuration items in the public configuration, as shown in the following figure.

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

## 2.2 Implementing Dataset Class

Pay attention to the following points when implementing the dataset:

1. Register the data class using `@ClassFactory.register(ClassType.DATASET)`.
2. Overload `_len___()` and `__getitem___()` for the dataloader.
3. Implement the `input_shape()` interface. The return value must correspond to the shape of the `__getitem__` data.

The code is as follows:

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

## 2.3 Commissioning

The preceding implementation can be directly used in the PipeStep in Vega or independently invoked. The code for independently invoking is as follows:

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

If the following information is displayed, the script is successfully executed:

```text
Ran 1 test in 12.119s

OK
```

## 2.4 Complete Code

For details about the complete code, see the following:

1. Dataset configuration: [cls_ds.py](https://github.com/huawei-noah/vega/blob/master/vega/datasets/conf/cls_ds.py)
2. Dataset Class implementation: [cls_ds.py](https://github.com/huawei-noah/vega/blob/master/vega/datasets/common/cls_ds.py)
