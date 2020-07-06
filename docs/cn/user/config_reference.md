# 配置参考

`Vega`高度模块化，通过配置可完成搜索空间、搜索算法、`pipeline`的构建，运行`Vega`应用就是加载配置文件，并根据配置来完成`AutoML`流程，如下：

```python
import vega


if __name__ == "__main__":
    vega.run("./main.yml")
```

以下是代码中`main.yml`配置文件中各个配置项的具体含义详细解释。

## 1. 整体结构

vega的配置可分为两部分：

1. 通用配置，配置项名称是`general`，用于设置公共和通用的一些配置项，如输出路径和日志级别等。
2. pipeline配置，包含两部分：
   1. pipeline定义，配置项名称是`pipeline`，是一个列表，包含了pipeline中的各个步骤。
   2. pipeline中各个步骤的定义，配置项名称是pipeline中定义的各个步骤名称。

```yaml
# 此处配置公共配置项，可参考随后的章节介绍
general:
    logger:
        level: info

# 定义pipeline。
pipeline: [my_nas, my_hpo, my_data_augmentation, my_fully_train]

# 定义每个步骤，可参考随后的章节介绍
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

以下详细介绍每个配置项。

## 2. 公共配置项

公共配置项中可以配置的配置项有：

| 配置项 | 说明 |
| :--: | :-- |
| local_base_path | 工作路径。每次系统运行，会在该路径下生成一个带有时间信息（我们称之为task id）的子文件夹，这样多次运行的输出不会被覆盖。在task id子文件夹下面一般包含output和worker两个子文件夹，output文件夹存储pipeline的每个步骤的输出数据，worker文件夹保存临时信息。 <br> **在集群的场景下，该路径需要设置为每个计算节点都可访问的EFS路径，用于不同节点共享数据。**|
| backup_base_path | 备份路径，这个设置主要用于云道环境，或者集群环境中，本地路径路径中的output和task会备份到该路径下。|
| timeout | worker超时时间，单位为小时，若在该时间范围内未完成，worker会被强制结束。单位为小时，缺省值为 10 个小时。|
| gpus_per_job | 搜索阶段每个worker使用的GPU数目，-1 代表一个worker使用该节点所有GPU，1 代表一个worker使用1个GPU，2 代表一个worker使用两个GPU，以此类推。|
| logger.level | 日志级别，可设置为：debug \| info \| warn \| error \| critical，缺省为 info。|
| cluster.master_ip | 在集群场景下需要设置该参数，设置为master节点的IP地址。 |
| cluster.listen_port | 在集群场景下需要关注该参数，若出现8000端口被占用，需要调整该监控端口。|
| cluster.slaves | 在集群场景下需要设置该参数，设置为除了master节点外的其他节点的IP地址。|

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

## 3. NAS配置项

NAS配置项一般包括如下内容：

| 配置项 | 说明 |
| :--: | :-- |
| pipe_step | 步骤类型，数值为 NasPipeStep。|
| search_algorithm | 搜索算法设置项，具体配置详细需要参考各个NAS算法。|
| search_space | 搜索空间定义，具体配置详见各个NAS算法。|
| trainer | trainer配置信息，请参考[Trainer 配置项](#trainer)。|
| dataset | 数据集配置，请参考[Dataet 配置项](#dataset)。|

如上提到的NAS算法包括：[Prune-EA](../algorithms/prune_ea.md), [Quant-EA](../algorithms/quant_ea.md), SM-NAS (开发中), [CARS](../algorithms/cars.md), [Segmentation-Adelaide-EA](../algorithms/Segmentation-Adelaide-EA-NAS.md), [SR-EA](../algorithms/sr-ea.md), [ESR-EA](../algorithms/sr-ea.md)

以下是BackboneNas算法的配置，供参考：

```yaml
my_nas:
    pipe_step:
        type: NasPipeStep
    search_algorithm:                   # 搜索算法设置项，Nas等步骤必须配置该项
        type: BackboneNas               # 搜索算法类型。可参见各个算法文档，了解所支持的搜索算法
        codec: BackboneNasCodec         # 编码器，一般和搜索算法对应
        policy:                        # 余下为搜索算法的配置项，请参考各个算法文档
            num_mutate: 10
            random_ratio: 0.2
        range:
            max_sample: 100
            min_sample: 10
    search_space:                       # 搜索空间定义，Nas等步骤必须配置该项
        type: SearchSpace
        modules: ['backbone', 'head']   # modules 是为了描述如何组合一个网络，请参考随后的描述
        backbone:                       # 每个module都对应了一个设置项，用于设置该项的参数
            ResNetVariant:              # 具体的网络配置项，可参考各个算法描述。
                base_depth: [18, 34, 50, 101]
                base_channel: [32, 48, 56, 64]
                doublechannel: [3, 4]
                downsample: [3, 4]
        head:
            LinearClassificationHead:
                num_classes: [10]
    trainer:                            # 配置训练器，必须配置项
        type: Trainer
    dataset:                            # dataset, 可参考dataset文档。
        type: Cifar10
```

其中配置项`search_space`的可选模型如下：

| module | 可选项 | 说明 | 算法参考 |
| :--: | :-- | :-- | :--: |
| backbone | PruneResNet | ResNet变种网络，用于支持剪枝操作。 | [参考](../algorithms/prune_ea.md) |
| backbone | QuantResNet | ResNet变种网络，用于支持量化操作。 | [参考](../algorithms/quant_ea.md) |
| backbone | ResNetVariant | ResNet变种网络，用于支持降采样点位调整等架构调整操作。 | [参考](../algorithms/sm-nas.md) |
| backbone | ResNet_Det | ResNet变种网络，用于目标检测任务的Backbone。 |  |
| head | LinearClassificationHead | 用于分类任务的网络分类层，可串接ResNetVariant。 |  |
| head | BBoxHead | 目标检测任务中的Bounding Box Head检测头。 |  |
| head | CurveLaneHead | 用于行道线检测的CurveLaneHead检测头。 |  |
| head | RPNHead | 目标检测任务中的Region Proposal Network Head检测头。 |  |
| neck | FPN | 目标检测任务中的Feature Pyramid Network。 |  |
| neck | FPN_CurveLane | 行道线检测任务中的Feature Pyramid Network。 |  |
| detector | FasterRCNN | 目标检测任务中的FasterRCNN检测网络。 |  |
| detector | AutoLaneDetector | 行道线检测任务中的AutoLaneDetector检测网络。 |  |
| super_network | DartsNetwork | Darts算法中的super network结构。 | [参考](../algorithms/cars.md) |
| super_network | CARSDartsNetwork | CARS算法中的super network结构。 | [参考](../algorithms/cars.md) |
| custom | AdelaideFastNAS | AdelaideFastNAS算法中自定义的网络结构。 | [参考](../algorithms/Segmentation-Adelaide-EA-NAS.md) |
| custom | MtMSR | MtMSR算法中自定义的网络结构。 | [参考](../algorithms/sr-ea.md) |

## 4. HPO配置项

HPO是指对模型训练运行参数的优化，不涉及到网络架构参数，可搜索项包括：

1. 数据集的batch size。
2. 优化方法，及其参数。
3. 学习率。
4. momentum。

HPO的配置项大概有如下部分：

| 配置项 | 说明 |
| :--: | :-- |
| pipe_step | 固定为HpoPipeStep |
| hpo | 配置超参信息，最主要的配置项是 type 和 hyperparameter_space。前者定义了使用哪种hpo算法，可参考[HPO算法文档](../algorithms/hpo.md)，后者定义了待搜索的超参信息。|
| trainer | trainer配置信息，请参考[Trainer 配置项](#trainer)。|
| dataset | 数据集配置，请参考[Dataet 配置项](#dataset)。|
| evaluator | evaluator信息, 请参考各个HPO算法的示例或Benchmark配置。 |

如上提到的HPO算法包括：[Prune-EA](../algorithms/prune_ea.md), [Quant-EA](../algorithms/quant_ea.md), SM-NAS (开发中), [CARS](../algorithms/cars.md), [Segmentation-Adelaide-EA](../algorithms/Segmentation-Adelaide-EA-NAS.md), [SR-EA](../algorithms/sr-ea.md), [ESR-EA](../algorithms/sr-ea.md)

如下是ASHA算法的HPO配置内容，供参考:

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

## 5. Data-Agumentation配置项

数据增广的配置主要有：

| 配置项 | 说明 |
| :--: | :-- |
| pipe_step | 固定为HpoPipeStep |
| hpo | 当前只支持PBA算法，固定为PBAHpo，可参考[PBA算法文档](../algorithms/pba.md)。|
| trainer | trainer配置信息，请参考[Trainer 配置项](#trainer)。|
| dataset | 数据集配置，请参考[Dataet 配置项](#dataset)。|
| evaluator | evaluator信息，请参考各个数据增广算法的介绍。 |

以下是PBA算法的配置信息，供参考：

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

## 6. Fully Train配置项

Fully Train用于训练网络模型，配置项如下：

| 配置项 | 说明 |
| :--: | :-- |
| pipe_step | 固定为FullyTrainPipeStep |
| models_folder | 待训练模型描述文件所在的目录，这个目录下的文件名格式为：model_desc_<ID>.json，其中ID为数字，这些模型会并行训练。这个选项和model选项互斥，且优先于model选项。 |
| trainer | trainer配置信息，请参考[Trainer 配置项](#trainer)。|
| dataset | 数据集配置，请参考[Dataet 配置项](#dataset)。|
| model | model信息，请参考[Trainer 配置项](#trainer)。|

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

## 7. Trainer 配置项

以上的每个pipeline的步骤中，都会有配置项trainer，可配置基本Trainer和扩展Trainer，trainer的基本配置信息有：

| 配置项 | 说明 |
| :--: | :-- |
| type | Trainer, 或算法扩展Trainer，可参考各个算法文档 |
| epochs | 总epochs数 |
| optim | 优化器及参数 |
| lr_scheduler | lr scheduler 及参数 |
| loss | loss 及参数 |
| metric | metric 及参数 |
| horovod | 是否启用horovod进行fully train, 启用horovod后，trainer会在horovod集群中使用所有计算资源训练指定的网络模型。若要启动horovod，必须要设置model选项。同时要注意数据集的shuffle参数设置为False。 |
| model_desc | 模型描述信息，和 model_desc_file 互斥，且model_desc_file优先。 |
| model_desc_file | 模型描述信息所在的文件，和 model_desc 互斥，且model_desc_file优先 |
| hps_file | 超参文件 |
| pretrained_model_file | 预训练模型文件 |

以下是加载torchvision的模型进行训练，示例如下：

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

如上例所示，除了可加载Vega定义的模型外，还可以加载 TorchVision Model，支持的模型如下，具体可参考[官方](https://pytorch.org/docs/stable/torchvision/models.html)描述。

| module | 可选项 |
| :--: | :-- |
| torch_vision_model | vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, squeezenet1_0, squeezenet1_1, shufflenetv2_x0.5, shufflenetv2_x1.0, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, mobilenet_v2, mnasnet0_5, mnasnet1_0, inception_v3_google, googlenet, densenet121, densenet169, densenet201, densenet161, alexnet, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_coco, keypointrcnn_resnet50_fpn_coco, maskrcnn_resnet50_fpn_coco, fcn_resnet101_coco, deeplabv3_resnet101_coco, r3d_18, mc3_18, r2plus1d_18 |

<span id="dataset"></span>

## 8. 数据集参考

每个pipeline都会涉及到数据集的配置，Vega将数据集分为三类: train, val, test，这三类可以独立配置，同时可以在Dataset中配置transform，以下为Cifar10数据集的配置示例：

```yaml
    dataset:
        type: Cifar10
        common:
            data_path: ~            # 配置数据集所在的目录。
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

Vega提供了常见数据集，如下：

| Name | Description | Data Source |
| :--: | --- | :--: |
| Cifar10 | The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images | [下载](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Cifar100 | The CIFAR-100 is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class | [下载](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Minist | The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples | [下载](http://yann.lecun.com/db/mnist/) |
| COCO | The COCO is a large-scale object detection, segmentation, and captioning dataset, about 123K images and 886K instances | [下载](http://cocodataset.org/#download) |
| Div2K | Div2K is a super-resolution architecture search database, containing 800 training images and 100 validiation images | [下载](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Imagenet | The ImageNet is an image database organized according to the WordNet hierarchy, in which each node of the hierarchy is depicted by hundreds and thousands of images | [下载](http://image-net.org/download-images) |
| Fmnist | Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.| [下载](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) |
| Cityscapes | The Cityscape is a large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames. | [下载](https://www.cityscapes-dataset.com/) |
| Cifar10TF | The CIFAR-10-bin dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images | [下载](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) |
| Div2kUnpair | DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and CVPR 2018) and @ PIRM (ECCV 2018)|[下载](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| ECP    | The ECP dataset. Focus on Persons in Urban Traffic Scenes.With over 238200 person instances manually labeled in over 47300 images, EuroCity Persons is nearly one order of magnitude larger than person datasets used previously for benchmarking.|[下载](https://eurocity-dataset.tudelft.nl/eval/downloads/detection) |

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

### 8.1 内置Transform

当前已支持如下Transform:

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
