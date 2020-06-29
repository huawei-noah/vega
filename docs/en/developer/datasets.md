# Dataset Development Guide

## 1. Introduction

The Vega provides interfaces and public methods related to data conversion and sampling in the Dataset class. The user data processing class can inherit from the Dataset class and use these public capabilities.

Vega provides common dataset classes, including Cifar10, Cifar100, ImageNet, Coco, FMnist, Mnist, Cityscapes, and Div2K. For details, see the  HYPERLINK "../user/config_reference.md"  Configuration Reference.Error! Hyperlink reference not valid.

### 1.1 Example

The following describes how to use the Dataset by using the cifar10 as an example. The procedure is as follows:

1. Adjust the default configuration. For example, change the location of the training set in the data file to a local file.

    ```yaml
    dataset:
        type: Cifar10
        train:
            data_path: "/data/dataset/"
     ```

2. In the program, use ClassFactory to create a Dataset,model to initialize the training set or test set, and use Dataloader to load data, as shown in the following figure.

    ```python
    dataset = ClassFactory.get_cls(Classtype.Dataset)
    train_data, test_data = dataset(model='train'), dataset(model='test')
    data_loader = train_data.dataloader
    for input, target in data_loader:
        process_data(input, target)
    ```

### 1.2 Architecture

All dataset classes of Vega are inherited from the base class Dataset. The base class Dataset defines the interfaces required by the dataset, provides attributes such as dataloader, transforms, and sampler, and provides default implementation. The derived class can reload these default implementations as required. The following describes how to customize a dataset.

## 2. Customize a dataset.

Assume that the user data contains 100 images and is stored in a folder. To implement a dataset class named MyDataset, perform the following steps:

1. Plan the data set.
2. Implements the dataloader.
3. Implement Transform.

As mentioned above, the MyDataset class inherits from Dataset as follows:

```python
from vega.datasets import Dataset
from vega.core.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.DATASET)
class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
```

In the preceding code, @ClassFactory.register(ClassType.DATASET) is used to register MyDataset with the Vega library.

## 2.1 Planning Data Set

A data set is divided into a training set and a test set. The training set is used to train a model, and the test set is used to verify the model. If all images in the example are used for training, you need to specify the data_path parameter for a file location.

During model training, data sets are dynamically divided into training sets and verification sets. You need to determine the sampling mode (sequential sampling or random sampling) and add a configuration parameter shuffle. The configuration information is as follows:

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

## 2.2 Implementing Dataloader

Assume that one image is loaded from the dataset each time, the image is loaded from the file each time, and the CV2 is used to load the image. The code is as follows:

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

## 2.3  Implementing Transform

Currently, Vega provides multiple [Transforms](../user/config_reference.md). 

Assume that MyDataset needs to implement the image flipping transform, the input is an original image, and the output is a flipped image. If Vega does not provide the transform, you need to call the flipping function of ImageOps to implement the transform. The code is as follows:

```python
import ImageOps


@TransformFactory.register()
class MyTransform():

    def __call__(self, img):
        return ImageOps.invert(img.convert('RGB'))
```

You only need to add the transform to the configuration file as follows:

```yaml
dataset:
    type: MyDataset
    train:
        data_path: "/data/dataset/"
        transforms:
            - type: MyTransform
```

If you need to adjust the transformations during model training, see HYPERLINK \l "transforms2" to adjust the transformations.Error! Hyperlink reference not valid.

## 2.5 Commissioning

The MyDataset class is newly implemented for commissioning. The code is as follows:

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

If the following information is displayed, the script is successfully executed:

```text
Ran 1 test in 12.119s

OK
```

## 2.6 Complete Code

Configuration file:

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

Code:

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

## 3. Reference

<span id=transform2></span>

1. Specify Transforms when initializing a dataset.

   ```python
    dataset = ClassFactory.get_cls(ClassType.DATASET)(
        mode="train",
        transforms=[tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip()]
        )
   ```

2. Dynamically adjusting Transforms during model training

 	The provides methods such as append(), insert(), remove(), and replace(). The methods are as follows:

    ```python
    dataset.transforms.append(tf.ToTensor())
    dataset.transforms.insert(2, "Color", level=2)
    dataset.transforms.remove("Color")
    dataset.transforms.replace(
        [tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip()]
        )
    ```
