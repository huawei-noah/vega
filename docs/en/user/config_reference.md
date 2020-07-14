# Configuration Reference

The Vega is highly modularized. The search space, search algorithm, and pipeline can be built through configuration. To run the Vega application is to load the configuration file and complete the AutoML process according to the configuration, as shown in the following figure.

```python
import vega


if __name__ == "__main__":
    vega.run("./main.yml")
```

The following describes the configuration items in the main.yml file in detail.

## 1. Overall structure

The configuration of the vega can be divided into two parts:

1. The general configuration item is used to set common and common configuration items, such as the output path and log level.
2. Pipeline configuration, including the following two parts:
   1. Pipeline definition. The configuration item name is pipeline, which is a list that contains all steps in the pipeline.
   2. Defines each step in Pipeline. The configuration item name is the name of each step defined in Pipeline.

```yaml
# Configure common configuration items.
general:
    logger:
        level: info

# Defining a Pipeline.
pipeline: [my_nas, my_hpo, my_data_augmentation, my_fully_train]

# defines each step. Refer to the following sections for details about
my_nas:
    pipe_step:
        type: NasPipeStep
    trainer:
        type: BackboneNasTrainer

my_hpo:
    pipe_step:
        type: HpoPipeStep
    trainer:
        type: Trainer

my_data_augmentation:
    pipe_step:
        type: HpoPipeStep
    trainer:
        type: Trainer

my_fully_train:
    pipe_step:
        type: FullyTrainPipeStep
    trainer:
        type: Trainer
```

The following describes each configuration item in detail.

## 2. Public configuration items

The following public configuration items can be configured:

| Configuration Item | Description |
| :--: | :-- |
| local_base_path | Working path. Each time when the system is running, a subfolder with time information (task id) is generated in the path. In this way, the output of multiple running is not overwritten. The task id subfolder contains two subfolders: output and worker. The output folder stores the output data of each step in the pipeline, and the worker folder stores temporary information.  <br> **In the clustered scenario, this path needs to be set to an EFS path that can be accessed by each computing node, and is used by different nodes to share data.**||
| backup_base_path | Backup path. This parameter is used in the cloud channel environment or cluster environment. The output and task files in the local path are backed up to this path. |
| timeout | Worker timeout interval, in hours. If the task is not completed within the interval, the worker is forcibly terminated. The unit is hour. The default value is 10. |
| gpus_per_job | Number of GPUs used by each worker in the search phase, -1 means that one worker uses all GPUs of the node, 1 means one worker uses one GPU, 2 means one worker uses two GPUs, and so on. |
| logger.level | Log level, which can be set to debug \| info \| warn \| error \| critical. default level is info. |
| cluster.master_ip | In the cluster scenario, this parameter needs to be set to the IP address of the master node. |
| cluster.listen_port | In the cluster scenario, you need to pay attention to this parameter. If port 8000 is occupied, you need to adjust the monitoring port. |
| cluster.slaves | In the cluster scenario, this parameter needs to be set to the IP address of other nodes except the master node. |

```yaml
general:
    task:
        local_base_path: "./tasks"
        backup_base_path: ~
    worker:
        timeout: 10.0
        gpus_per_job: -1
    logger:
        level: info
    cluster:
        master_ip: ~
        listen_port: 8000
        slaves: []
```

## 3. NAS configuration items

NAS configuration items include:

| Configuration Item | Description |
| :--: | :-- |
| pipe_step | Step type. The value is fixed to NasPipeStep. |
| search_algorithm | Search algorithm configuration item. For details, see the configuration of each NAS algorithm. |
| search_space | For details about the definition of the search space, see each NAS algorithm. |
| trainer | Trainer configuration information. For details, see the [Trainer Configuration](#trainer). |
| dataset | Dataset configuration. For details, see the [Dataset Configuration](#dataset). |

The NAS algorithm mentioned above includes: [Prune-EA](../algorithms/prune_ea.md), [Quant-EA](../algorithms/quant_ea.md),SM-NAS (Coming soon), [CARS](../algorithms/cars.md), [Segmentation-Adelaide-EA](../algorithms/Segmentation-Adelaide-EA-NAS.md), [SR-EA](../algorithms/sr-ea.md), [ESR-EA](../algorithms/sr-ea.md)

The following is the configuration of the BackboneNas algorithm:

```yaml
my_nas:
    pipe_step:
        type: NasPipeStep
    search_algorithm:       # search algorithm configuration item. This item must be configured in steps such as NAS.
        type: BackboneNas   # Search algorithm type.
        codec: BackboneNasCodec # The supported search algorithm codec is BackboneNas
        policy:                 # For details about the supported search algorithms, see the related algorithm documents.
            num_mutate: 10
            random_ratio: 0.2
        range:
            max_sample: 100
            min_sample: 10
    search_space:                       # search space in the related algorithm document, this item must be configured in steps such as Nas.
        type: SearchSpace
        modules: ['backbone', 'head']   # Modules are used to describe how to combine a network.
        backbone:                       # Each module has a configuration item
            ResNetVariant:              # For details, see the description of each algorithm. 
                base_depth: [18, 34, 50, 101]
                base_channel: [32, 48, 56, 64]
                doublechannel: [3, 4]
                downsample: [3, 4]
        head:
            LinearClassificationHead:
                num_classes: [10]
    trainer:
        type: Trainer
    dataset:
        type: Cifar10
```

The optional models of the search_space configuration item are as follows:

| module | Optional | Description | Algorithm Reference |
| :--: | :-- | :-- | :--: |
| backbone | PruneResNet | ResNet variant network, which is used to support the prune operation. | [ref](../algorithms/prune_ea.md) |
| backbone | QuantResNet | ResNet variant network, which is used to support quantization operations. | [ref](../algorithms/quant_ea.md) |
| backbone | ResNetVariant | The ResNet variant network is used to support architecture adjustment operations such as down-sampling point adjustment. | [ref](../algorithms/sm-nas.md) |
| backbone | ResNet_Det | Indicates the ResNet variant network, which is used for the backbone of the target detection task. |  |
| head | LinearClassificationHead | Network classification layer used to classify tasks, which can be concatenated with ResNetVariant. |  |
| head | BBoxHead | Bounding box head in the object detection task. |  |
| head | CurveLaneHead | The CurveLaneHead detection head is used to detect the lane. |  |
| head | RPNHead | Region Proposal Network Head in the object detection task. |  |
| neck | FPN | Indicates the feature deployed network in the object detection task. |  |
| neck | FPN_CurveLane | Indicates the feature dashamid network in the roadway detection task. |  |
| detector | FasterRCNN | Faster R-CNN detection network in the object detection task. |  |
| detector | AutoLaneDetector | AutoLaneDetector detection network in the roadway detection task. |  |
| super_network | DartsNetwork | Super network structure in the Darts algorithm. | [ref](../algorithms/cars.md) |
| super_network | CARSDartsNetwork | Super network structure in the CARS algorithm. | [ref](../algorithms/cars.md) |
| custom | AdelaideFastNAS | Indicates the user-defined network structure in the AdelaideFastNAS algorithm. | [ref](../algorithms/Segmentation-Adelaide-EA-NAS.md) |
| custom | MtMSR | Indicates the user-defined network structure in the MtMSR algorithm. | [ref](../algorithms/sr-ea.md) |

## 4. HPO configuration items

HPO refers to the optimization of model training running parameters. It does not involve network architecture parameters. The searchable items are as follows:

1. Batch size of the dataset.
2. Optimization method and related parameters.
3. Learning rate.
4. Momentum.

The HPO configuration items are as follows:

| Configuration Item | Description |
| :--: | :-- |
| pipe_step | The value is fixed at HpoPipeStep. |
| hpo | Configure the type and domain_space parameters. The former defines the HPO algorithm to be used. For details, see the [HPO](../algorithms/hpo.md). The latter defines the hyperparameter information to be searched for. |
| trainer | Trainer configuration information. For details, see the [Trainer Configuration](#trainer). |
| dataset | Dataset configuration. For details, see the [Dataset Configuration](#dataset). |
| evaluator | evaluator information. Please refer to each HPO algorithm example or Benchmark configuration. |
The HPO configuration of the ASHA algorithm is as follows for reference:

```yaml
my_hpo:
    pipe_step:
        type: HpoPipeStep
    hpo:
        type: AshaHpo
        policy:
            total_epochs: 81
            config_count: 40
        hyperparameter_space:
            hyperparameters:
                -   key: dataset.batch_size
                    type: INT_CAT
                    range: [8, 16, 32, 64, 128, 256]
                -   key: trainer.optim.lr
                    type: FLOAT_EXP
                    range: [0.00001, 0.1]
                -   key: trainer.optim.type
                    type: STRING
                    range: ['Adam', 'SGD']
                -   key: trainer.optim.momentum
                    type: FLOAT
                    range: [0.0, 0.99]
            condition:
                -   key: condition_for_sgd_momentum
                    child: trainer.optim.momentum
                    parent: trainer.optim.type
                    type: EQUAL
                    range: ["SGD"]
    model:
        model_desc:
            modules: ["backbone", "head"]
            backbone:
                base_channel: 64
                downsample: [0, 0, 1, 0, 1, 0, 1, 0]
                base_depth: 18
                doublechannel: [0, 0, 1, 0, 1, 0, 1, 0]
                name: ResNetVariant
            head:
                num_classes: 10
                name: LinearClassificationHead
                base_channel: 512
    dataset:
        type: Cifar10
    trainer:
        type: Trainer
    evaluator:
        type: Evaluator
        gpu_evaluator:
            type: GpuEvaluator
            ref: trainer
```

## 5. Data-Agumentation configuration item

The configuration of data augmentation includes:

| Configuration Item | Description |
| :--: | :-- |
| pipe_step | The value is fixed at HpoPipeStep. |
| hpo | Currently, only the  HYPERLINK "../algorithms/pba.md" PBA algorithm is supported. The value is fixed to PBAHpo. For details, see the [PBA](../algorithms/pba.md). |
| trainer | Trainer configuration information. For details, see the [Trainer Configuration](#trainer). |
| dataset | Dataset configuration. For details, see the [Dataset Configuration](#dataset). |

The following shows the configuration of the PBA algorithm for reference:

```yaml
my_data_augmentation:
    pipe_step:
        type: HpoPipeStep
    dataset:
        type: Cifar10
    hpo:
        type: PBAHpo
        each_epochs: 3
        config_count: 16
        total_rungs: 200
        transformers:
            Cutout: True
            Rotate: True
            Translate_X: True
            Translate_Y: True
            Brightness: True
            Color: True
            Invert: True
            Sharpness: True
            Posterize: True
            Shear_X: True
            Solarize: True
            Shear_Y: True
            Equalize: True
            AutoContrast: True
            Contrast: True
    trainer:
        type: Trainer
    evaluator:
        type: Evaluator
        gpu_evaluator:
            type: GpuEvaluator
```

## 6. Fully Train Configuration

Full training is used to train network models. The configuration items are as follows:

| Configuration Item | Description |
| :--: | :-- |
| pipe_step | The value is fixed at FullyTrainPipeStep.  |
| models_folder | The directory where the model description file to be trained is located. The file name format in this directory is: model_desc_\<ID>.json, where the ID is a number, and these models will be trained in parallel. This option is mutually exclusive  with the parameter "model" and has priority to "model". |
| trainer | Trainer configuration information. For details, see the [Trainer Configuration](#trainer). |
| dataset | Dataset configuration. For details, see the [Dataset Configuration](#dataset). |
| model | Model information. For details, see the [Trainer Configuration](#trainer). |
| model_desc_file | The model description file |

```yaml
my_fully_train:
    pipe_step:
        type: FullyTrainPipeStep
        # models_folder: ~
    trainer:
        type: Trainer
    model:
        model_desc_file: "/models/model_desc.json"
    dataset:
        type: Cifar10
```

<span id="trainer"></span>

## 7. Trainer configuration item

In each of the preceding pipeline steps, the configuration item trainer is provided. You can configure the basic trainer and extended trainer. The basic configuration information of the trainer is as follows:

| Configuration Item | Description |
| :--: | :-- |
| type | Trainer, or algorithm extension trainer. For details, see the related algorithm document. |
| epochs | Total epochs |
| optim | Optimizers and Parameters |
| lr_scheduler | lr scheduler and parameters |
| loss | Loss and Parameters |
| metric | Metrics and Parameters |
| horovod | Whether to enable Horovod for fully train. After enabling Horovod, the trainer will use all computing resources in the Horovod cluster to train the specified network model. To start horovod, you must set the `model` option. |
| model_desc | Model description, which is mutually exclusive with model_desc_file. model_desc_file takes precedence over model_desc_file. And the parameter shuffle of the dataset must be set to False. |
| model_desc_file | File where the model description information is located. This parameter is mutually exclusive with model_desc, and model_desc_file takes priority over model_desc_file. |
| hps_file | Hyper-parameter file |
| pretrained_model_file | Pre-trained model file |

The following is an example of loading the Torchvision model for training:

```yaml
    trainer:
        type: Trainer
        epochs: 160
        optim:
            type: Adam
            lr: 0.1
        lr_scheduler:
            type: MultiStepLR
            milestones: [75, 150]
            gamma: 0.5
        metric:
            type: accuracy
        loss:
            type: CrossEntropyLoss
        horovod: False
    dataset:
        type: Imagenet
    model:
        model_desc:
            modules: ['backbone', 'head']
            backbone:
                ResNetVariant:
                    base_depth: [18, 34, 50, 101]
                    base_channel: [32, 48, 56, 64]
                    doublechannel: [3, 4]
                    downsample: [3, 4]
            head:
                LinearClassificationHead:
                    num_classes: [10]
        # model_desc_file: ~
        # hps_file: ~
        # pretrained_model_file: ~
```

As shown in the preceding example, in addition to the models defined by Vega, you can also load the TorchVision Model. The following models are supported. For details, see the [official desc](https://pytorch.org/docs/stable/torchvision/models.html).

| module | Optional |
| :--: | :-- |
| torch_vision_model | vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, squeezenet1_0, squeezenet1_1, shufflenetv2_x0.5, shufflenetv2_x1.0, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, mobilenet_v2, mnasnet0_5, mnasnet1_0, inception_v3_google, googlenet, densenet121, densenet169, densenet201, densenet161, alexnet, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_coco, keypointrcnn_resnet50_fpn_coco, maskrcnn_resnet50_fpn_coco, fcn_resnet101_coco, deeplabv3_resnet101_coco, r3d_18, mc3_18, r2plus1d_18 |

<span id="dataset"></span>

## 8. Dataset Reference

Each pipeline involves dataset configuration. Vega classifies datasets into three types: train, val, and test. The three types of datasets can be configured independently. In addition, transform can be configured in Dataset. The following is a configuration example of the Cifar10 dataset:

```yaml
    dataset:
        type: Cifar10
        common:
            data_path: ~            # configuration data set is located.
            batch_size: 256
            num_workers: 4
            imgs_per_gpu: 1
            train_portion: 0.5
            shuffle: false
            distributed: false
        train:
            transforms:
                - type: RandomCrop
                size: 32
                padding: 4
                - type: RandomHorizontalFlip
                - type: ToTensor
                - type: Normalize
                mean:
                    - 0.49139968
                    - 0.48215827
                    - 0.44653124
                std:
                    - 0.24703233
                    - 0.24348505
                    - 0.26158768
        val:
            transforms:
                - type: ToTensor
                - type: Normalize
                mean:
                    - 0.49139968
                    - 0.48215827
                    - 0.44653124
                std:
                    - 0.24703233
                    - 0.24348505
                    - 0.26158768
        test:
            transforms:
                - type: ToTensor
                - type: Normalize
                mean:
                    - 0.49139968
                    - 0.48215827
                    - 0.44653124
                std:
                    - 0.24703233
                    - 0.24348505
                    - 0.26158768
```

### 8.1 内置数据集

Vega provides the following common data sets:

| Name | Description | Data Source |
| :--: | --- | :--: |
| Cifar10 | The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images | [download](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Cifar100 | The CIFAR-100 is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class | [download](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Minist | The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples | [download](http://yann.lecun.com/db/mnist/) |
| COCO | The COCO is a large-scale object detection, segmentation, and captioning dataset, about 123K images and 886K instances | [download](http://cocodataset.org/#download) |
| Div2K | Div2K is a super-resolution architecture search database, containing 800 training images and 100 validiation images | [download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Imagenet | The ImageNet is an image database organized according to the WordNet hierarchy, in which each node of the hierarchy is depicted by hundreds and thousands of images | [download](http://image-net.org/download-images) |
| Fmnist | Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.| [download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) |
| Cityscapes | The Cityscape is a large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames. | [download](https://www.cityscapes-dataset.com/) |
| Cifar10TF | The CIFAR-10-bin dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images | [download](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) |
| Div2kUnpair | DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and CVPR 2018) and @ PIRM (ECCV 2018)|[download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| ECP    | The ECP dataset. Focus on Persons in Urban Traffic Scenes.With over 238200 person instances manually labeled in over 47300 images, EuroCity Persons is nearly one order of magnitude larger than person datasets used previously for benchmarking.|[download](https://eurocity-dataset.tudelft.nl/eval/downloads/detection) |

1. Cifar10 Default Configuration

    ```yaml
    data_path: ~            # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/cifar10
    batch_size: 256         # batch size
    num_workers: 4          # the worker number to load the data
    shuffle: false          # if True, will shuffle, defaults to False
    distributed: false      # whether to use distributed train
    imgs_per_gpu: 1         # image number per gpu
    train_portion: 0.5      # the ratio of the train data split from the initial train data
    ```

2. Cifar100 Default Configuration

    ```yaml
    data_path: ~            # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/cifar100
    batch_size: 1           # batch size
    num_workers: 4          # the worker number to load the data
    shuffle: true           # if True, will shuffle, defaults to False
    distributed: false      # whether to use distributed train
    imgs_per_gpu: 1         # image number per gpu
    ```

3. Cityscapes Default Configuration

    ```yaml
    root_path: ~            # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/Cityscapes
    list_path: 'img_gt_train.txt'   # the name of the txt file
    batch_size: 1           # batch size
    mean: 128               # the parameter mean for transform
    ignore_label: 255       # the label to ignore
    scale: True             # if scale is true, the asptio will be keep when transform
    mirrow: True            # whether to use mirrow for transform  
    rotation: 90            # the rotation value
    crop: 321               # the crop size
    num_workers: 4          # the worker number to load the data
    shuffle: False          # if True, will shuffle, defaults to False
    distributed: True       # whether to use distributed train
    id_to_trainid: False    # change the random id to continious id,if true, a dict should be obtain
    imgs_per_gpu: 1         # image number per gpu
    ```

4. DIV2K Default Configuration

    ```yaml
    root_HR: ~              # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/DIV2K/div2k_train/hr
    root_LR: ~              # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/DIV2k/div2k_train/lr
    batch_size: 1           # batch size
    num_workers: 4          # the worker number to load the data
    upscale: 2              # the upscale for super resolution
    subfile: !!null         # whether to use subfile,Set it to None by default
    crop: !!null            # the crop size,Set it to None by default
    shuffle: false          # if True, will shuffle, defaults to False
    hflip: false            # whether to use horrizion flip
    vflip: false            # whether to use vertical flip
    rot90: false            # whether to use rotation
    distributed: True       # whether to use distributed train
    imgs_per_gpu: 1         # image number per gpu
    ```

5. FashionMnist Default Configuration

    ```yaml
    data_path: ~            # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/fmnist
    batch_size: 1           # batch size
    num_workers: 4          # the worker number to load the data
    shuffle: true           # if True, will shuffle, defaults to False
    distributed: false      # whether to use distributed train
    imgs_per_gpu: 1         # image number per gpu
    ```

6. Imagenet Default Configuration

    ```yaml
    data_path: ~            #  the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/ImageNet
    batch_size: 1           #  batch size
    num_workers: 4          #  the worker number to load the data
    shuffle: true           #  if True, will shuffle, defaults to False
    distributed: false      #  whether to use distributed train
    imgs_per_gpu: 1         #  image number per gpu
    ```

7. Mnist Default Configuration

    ```yaml
    data_path: ~            # the path of the dataset, default is None, MUST be set to a correct dataset PATH, such as /datasets/mnist
    batch_size: 1           # batch size
    num_workers: 4          # the worker number to load the data
    shuffle: true           # if True, will shuffle, defaults to False
    distributed: false      # whether to use distributed train
    imgs_per_gpu: 1         # image number per gpu
    ```

### 8.1 Built-in Transform

Currently, the following transforms are supported:

| Transform | Input | Output |
| --- | --- | --- |
| AutoContrast | level img | img |
| BboxTransform | bboxes imge_shape scale_factor | bboxes |
| Brightness | level img | img |
| Color | level img | img |
| Contrast | level img | img |
| Cutout | length img | img |
| Equalize | level img | img |
| ImageTransform | scale img | img img_shape pad_shape scale_factor |
| Invert | level img | img |
| MaskTransform | masks pad_shape scale_factor | padded_masks |
| Numpy2Tensor | numpy | tensor |
| Posterize | level img | img |
| RandomCrop_pair | crop upscale img label | img label |
| RandomHorizontalFlip_pair | img label | img label |
| RandomMirrow_pair | img label | img label |
| RandomRotate90_pair | img label | img label |
| RandomVerticallFlip_pair | img label | img label |
| Rotate | level img | img |
| SegMapTransform | scale img | img |
| Sharpness | level img | img |
| Shear_X | level img | img |
| Shear_Y | level img | img |
| Solarize | level img | img |
| ToPILImage_pair | img1 img2 | img1 img2 |
| ToTensor_pair | img1 img2 | tensor1 tensor2 |
| Translate_X | level img | img |
| Translate_Y | level img | img |
