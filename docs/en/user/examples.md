# Examples Reference

Vega provides guidance on how to use algorithms and tasks, and provides guidance on algorithm development for developers, such as expanding the search space and search algorithm and building datasets applicable to Vega.

## 1. Example List

In the following example, the following directory is generated after the package is decompressed:

| Contents | Description |
| :--: | :-- |
| compression | compression algorithm usage example, including [Quant-EA](../algorithms/quant_ea.md)、 [Prune-EA](../algorithms/prune_ea.md). |
| data augmentation | Example of using the data augmentation algorithm, including the [PBA](../algorithms/pba.md), [CycleSR](../algorithms/cyclesr.md) |
| hpo | hyperparameter optimization algorithm use examples, including  [ASHA](../algorithms/hpo.md), [BO](../algorithms/hpo.md), [TPE](../algorithms/hpo.md), [BOHB](../algorithms/hpo.md), [BOSS](../algorithms/hpo.md), Random, Random Pareto. |
| nas | Examples of network architecture search, including [CARS](../algorithms/cars.md), [SP-NAS](../algorithms/sp_nas.md), [auto-lane](../algorithms/auto_lane.md), [SR-EA](../algorithms/sr_ea.md), [ESR-EA](../algorithms/esr_ea.md), [Adelaide-EA](../algorithms/adelaide_ea.md), [NAGO](../algorithms/nago.md), BackboneNas, DartsCNN, FIS, GDAS, MFKD, SegmentationEA, SGAS, ModuleNas, DNet-NAS |
| fully train | Samples related to fully train, including the EfficientNet B0/B4 model training example and FMD operator example. |
| classification | An example of using NAS + HPO + FullyTrain to complete an image classification task |
| features | Cluster, custom dataset, model evaluation, quota samples. |

## 2. Run Example

### 2.1 Run PyTorch Example

Generally, an algorithm example contains a configuration file, some algorithms, and some matching code.

You can execute the following commands in the `examples` directory:

```bash
vega <algorithm config file>
```

For example, the command of CARS example is as follows:

```bash
vega ./nas/cars/cars.yml
```

All information is stored in configuration files. Configuration items are classified into public configuration items and algorithm configuration items. For details about public configuration items, see the [configuration reference](./config_reference.md). For details about algorithm configuration, see the reference documents of each algorithm.Error! Hyperlink reference not valid.

Before running an example, you need to configure the directory where the dataset is located in the algorithm configuration file. The root directory of the default dataset is `/cache/datasets/`, for example, the directory of the `Cifar10` is `/cache/datasets/cifar10/`.

Before running the example, you need to download the dataset to the default data configuration directory. Before running the example, you need to create the directory `/cache/datasets/`, then download each dataset to the directory and unzip it. The default directory configuration of each dataset is as follows:

| example | Pre-trained Model | Default Path | Model Source |
| :--: | :-- | :-- | :--: |
| adelaide_ea | mobilenet_v2-b0353104.pth | /cache/models/mobilenet_v2-b0353104.pth | [download](https://box.saas.huaweicloud.com/p/e9e06f49505a1959da6cba9401b2bf38) |
| BackboneNas (mindspore) | resnet50-19c8e357.pth | /cache/models/resnet50-19c8e357.pth | [download]() |
| BackboneNas (tensorflow), classification, prune_ea(tensorflow) | resnet_imagenet_v1_fp32_20181001 | /cache/models/resnet_imagenet_v1_fp32_20181001/ <br>  keep only these files: checkpoint, graph.pbtxt, model.ckpt-225207.data-00000-of-00002, model.ckpt-225207.data-00001-of-00002, model.ckpt-225207.index, model.ckpt-225207.meta | [download](http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v1_fp32_20181001.tar.gz) |
| dnet_nas | 031-_64_12-1111-11211112-2.pth | /cache/models/031-_64_12-1111-11211112-2.pth | [download]() |
| prune_ea(pytorch) | resnet20.pth | /cache/models/resnet20.pth | [download](https://box.saas.huaweicloud.com/p/67cd96e5da41b1c5a88f2b323446c0f8) |
| prune_ea(mindspore) | resnet20.ckpt | /cache/models/resnet20.ckpt | [download](https://box.saas.huaweicloud.com/p/7f1743a041a0ede7f68713d1360a57d5) |
| sp_nas | fasterrcnn_coco.pth | /cache/models/fasterrcnn_coco.pth | [download]() |
| sp_nas | fasterrcnn_serialnet_backbone.pth | /cache/models/fasterrcnn_serialnet_backbone.pth | [download]() |
| sp_nas | serial_classification_net.pth | /cache/models/serial_classification_net.pth | [download]() |
| sp_nas | torch_fpn.pth | /cache/models/torch_fpn.pth | [download]() |
| sp_nas | torch_rpn.pth | /cache/models/torch_rpn.pth | [download]() |

In the configuration file of each example, the platform description (PyTorch, TensorFlow, and MindSpore) applicable to the example is provided in `general/backend`.

For example, the following configuration indicates that the sample can run on three platforms:

```yaml
general:
    backend: pytorch # pytorch | tensorflow | mindspore
```

The following configurations can run only in TensorFlow:

```yaml
general:
    backend: tensorflow
```

### 2.2 Run TensorFlow Example

1. Command line (GPU):

    ```bash
    vega <algorithm config file> -b t
    ```

    for example:

    ```bash
    vega ./nas/backbone_nas/backbone_nas.yml -b t
    ```

2. Command line (Atlas 900):

    ```bash
    vega <algorithm config file> -b t -d NPU
    ```

    for example:

    ```bash
    vega ./nas/backbone_nas/backbone_nas.yml -b t -d NPU
    ```

### 2.3 Run MindSpore Example

Command line (Atlas 900):

```bash
vega <algorithm config file> -b m -d NPU
```

for example:

```bash
vega ./nas/backbone_nas/backbone_nas.yml -b m -d NPU
```

## 3. Examples' Input and Output

### 3.1 Model Compression

1. Prune-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: compression/prune-ea/prune.yml <br> Pre-Trained Model: /cache/models/resnet20.pth <br> Dataset: /cache/datasets/cifar10 |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | nas | approximate running time | (random_models + num_generation * num_individual) * epochs / Number of GPUs * Training time per epoch |
    | fully train | Input | Config File: compression/prune-ea/prune.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

2. Quant-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: compression/quant-ea/quant.yml <br> Dataset: /cache/datasets/cifar10 |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | nas | approximate running time | (random_models + num_generation * num_individual) * epochs / Number of GPUs * Training time per epoch |
    | fully train | Input | Config File: compression/quant-ea/quant.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

### 3.2 NAS

1. CARS

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: nas/cars/cars.yml <br> Dataset: /cache/datasets/cifar10 |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | nas | approximate running time | epochs * Training time per epoch (The training time is affected by num_individual) |
    | fully train | Input | Config File: nas/cars/cars.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

2. Adelaide-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | random | Input | Config File: nas/adelaide_ea/adelaide_ea.yml <br> Dataset: /cache/datasets/cityscapes |
    | random | Output | Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json |
    | random | approximate running time | max_sample * epochs / Number of GPUs * Training time per epoch |
    | mutate | Input | Config File: nas/adelaide_ea/adelaide_ea.yml <br> Dataset: /cache/datasets/cityscapes <br> Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json  |
    | mutate | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json |
    | mutate | approximate running time | max_sample * epochs / Number of GPUs * Training time per epoch |
    | fully train | Input | Config File: nas/adelaide_ea/adelaide_ea.yml <br> Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cityscapes |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

3. ESR-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: nas/esr_ea/esr_ea.yml <br> Dataset: /cache/datasets/DIV2K |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/selected_arch.npy |
    | nas | approximate running time | num_generation * num_individual * epochs / Number of GPUs * Training time per epoch |
    | fully train | Input | Config File: nas/esr_ea/esr_ea.yml <br> Network Description File: tasks/\<task id\>/output/nas/selected_arch.npy <br> Dataset: /cache/datasets/DIV2K |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

4. SR-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | random | Input | Config File: nas/sr_ea/sr_ea.yml <br> Dataset: /cache/datasets/DIV2K |
    | random | Output | Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json |
    | random | approximate running time | num_sample * epochs / Number of GPUs * Training time per epoch |
    | mutate | Input | Config File: nas/sr_ea/sr_ea.yml <br> Dataset: /cache/datasets/DIV2K <br> Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json  |
    | mutate | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json |
    | mutate | approximate running time | num_sample * epochs / Number of GPUs * Training time per epoch |
    | fully train | Input | Config File: nas/sr_ea/sr_ea.yml <br> Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Dataset: /cache/datasets/DIV2K |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

5. SP-NAS

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas1 | Input | Config File: nas/sp_nas/spnas.yml <br> Dataset: /cache/datasets/COCO2017 <br> Pre-Trained Model: /cache/models/resnet50-19c8e357.pth <br> Config File:  nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | nas1 | Output | Network Description File: tasks/\<task id\>/output/nas1/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/total_list_p.csv |
    | nas1 | approximate running time | max_sample * epochs / Number of GPUs * Training time per epoch |
    | nas2 | Input | Config File: nas/sp_nas/spnas.yml <br> Dataset: /cache/datasets/COCO2017 <br> Network Description File: tasks/\<task id\>/output/nas1/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/total_list_p.csv <br> Config File:  nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | nas2 | Output | Network Description File: tasks/\<task id\>/output/nas2/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/total_list_s.csv |
    | nas2 | approximate running time | max_sample * epochs / Number of GPUs * Training time per epoch |
    | fully train | Input |  Config File: nas/sp_nas/spnas.yml <br> Dataset: /cache/datasets/COCO2017 <br> Network Description File: tasks/\<task id\>/output/nas2/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/total_list_s.csv <br> Config File:  nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | fully train | Output | Model: tasks/\<task id\>/output/fullytrain/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

6. Auto_Lane

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    |nas|Input|Config File：nas/auto_lane/auto_lane.yml <br/> Dataset：/cache/datasets/CULane  OR  /cache/datasets/CurveLane|
    |nas|Output|Network Description File：tasks/\<task id\>/output/nas/model_desc_\<id\>.json|
    |nas|approximate running time|max_sample * epoch / Numbers of GPUs * Training time per epoch|
    |fully train|Input|Config File：nas/sp_nas/auto_lane.yml <br> Dataset：/cache/datasets/CULane  OR      /cache/datasets/CurveLane<br> Network Description File：tasks/\<task id\>/output/nas/model_desc_\<id\>.json|
    |fully train|Output|Model：tasks/\<task id\>/output/fullytrain/model_\<id\>.pth|
    |fully train|approximate running time|epochs * Training time per epoch|

7. AutoGroup

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File nas/fis/autogroup.yml <br> Dataset: /cache/datasets/avazu |
    | fully train | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Model: tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

8. AutoFis

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | search | Input | Config File nas/fis/autogate_grda.yml <br> Dataset: /cache/datasets/avazu |
    | search | Output | Model: tasks/\<task id\>/output/search/0/model.pth |
    | search | approximate running time | epochs * Training time per epoch |
    | retrain | Input | Config File nas/fis/autogate_grda.yml <br> Dataset: /cache/datasets/avazu |
    | retrain | Output | Model: tasks/\<task id\>/output/retrain/0/model.pth |
    | retrain | approximate running time | epochs * Training time per epoch | 

### 3.3 Data Augmentation

1. PBA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | pba | Input | Config File: data_augmentation/pba/pba.yml <br> Dataset: /cache/datasets/cifar10 |
    | pba | Output | Transformer List: tasks/\<task id\>/output/pba/best_hps.json |
    | pba | approximate running time | total_rungs * each_epochs * config_count / Number of GPUs * Training time per epoch |

2. CycleSR

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File: data_augmentation/cyclesr/cyclesr.yml <br> Dataset: /cache/datasets/DIV2K_unpair |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | approximate running time | n_epoch * Training time per epoch |

### 3.4 HPO

1. ASHA、BOHB、BOSS

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | hpo | Input | Config File: hpo/asha\|bohb\|boss/hpo/asha\|bohb\|boss.yml <br> Dataset: /cache/datasets/cifar10 |
    | hpo | Output | Hyperparameter file: tasks/\<task id\>/output/hpo/best_hps.json |

### 3.5 Fully Train

1. EfficientNet

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File: fully_train/efficientnet/efficientnet_b0.yml <br> Dataset: /cache/datasets/ILSVRC |
    | fully train | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

2. FMD

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File: fully_train/fmd/fmd.yml <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Model: tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | approximate running time | epochs * Training time per epoch |

3. ResNet

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File fully_train/trainer/resnet.yml <br> Dataset: /cache/datasets/ILSVRC |
    | fully train | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Model: tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | approximate running time | epochs * Training time per epoch |
