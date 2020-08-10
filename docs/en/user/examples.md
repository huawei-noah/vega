# Examples Reference

Vega provides guidance on how to use algorithms and tasks, and provides guidance on algorithm development for developers, such as expanding the search space and search algorithm and building datasets applicable to Vega.

## 1. Download Example

In the following example, the following directory is generated after the package is decompressed:

| Contents | Description |
| :--: | :-- |
| compression | compression algorithm usage example, including [Quant-EA](../algorithms/quant_ea.md)、 [Prune-EA](../algorithms/prune_ea.md). |
| data augmentation | Example of using the data augmentation algorithm, including the [PBA](../algorithms/pba.md). |
| hpo | hyperparameter optimization algorithm use examples, including  [ASHA](../algorithms/hpo.md), [BO](../algorithms/hpo.md), [TPE](../algorithms/hpo.md), [BOHB](../algorithms/hpo.md), [BOSS](../algorithms/hpo.md). |
| nas | Examples of network architecture search, including [SM-NAS](../algorithms/sm-nas.md), [CARS](../algorithms/cars.md), [SP-NAS](../algorithms/sp-nas.md), [auto-lane](../algorithms/auto_lane.md), [SR-EA](../algorithms/sr-ea.md), [ESR-EA](../algorithms/esr_ea.md), [Adelaide-EA](../algorithms/Segmentation-Adelaide-EA-NAS.md) |
| searchspace | [fine grained search space](../developer/fine_grained_search_space.md) |
| fully train | Samples related to fully train, including RESTNET18 and CARS models for training the torch model. |
| tasks/classification | An example of using NAS + HPO + FullyTrain to complete an image classification task |

## 2. Example Description

Generally, an algorithm example contains a configuration file, some algorithms, and some matching code.

You can execute the following commands in the `examples` directory:

```bash
python3 ./run_example.py <algorithm config file>
```

For example, the command of CARS example is as follows:

```bash
python3 ./run_example.py ./nas/cars/cars.yml
```

All information is stored in configuration files. Configuration items are classified into public configuration items and algorithm configuration items. For details about public configuration items, see the [configuration reference](./config_reference.md). For details about algorithm configuration, see the reference documents of each algorithm.Error! Hyperlink reference not valid.

Before running an example, you need to configure the directory where the dataset is located in the algorithm configuration file. The root directory of the default dataset is `/cache/datasets/`, for example, the directory of the `Cifar10` is `/cache/datasets/cifar10/`.

Before running the example, you need to download the dataset to the default data configuration directory. Before running the example, you need to create the directory `/cache/datasets/`, then download each dataset to the directory and unzip it. The default directory configuration of each dataset is as follows:

| Dataset | Default Path | Data Source | Note |
| :--- | :--- | :--: | :-- |
| Cifar10 | /cache/datasets/cifar10/ | [Download](https://www.cs.toronto.edu/~kriz/cifar.html) | |
| Cifar10TF | /cache/datasets/cifar-10-batches-bin/ | [Download](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) | |
| ImageNet | /cache/datasets/ILSVRC/ | [Download](http://image-net.org/download-images) | |
| ImageNetTF | /cache/datasets/imagenet_tfrecord/ | [Download](http://image-net.org/download-images) | **Use [code](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) to convert data** |
| COCO | /cache/datasets/COCO2017 | [Download](http://cocodataset.org/#download) | |
| Div2K | /cache/datasets/DIV2K/ | [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | |
| Div2kUnpair | /cache/datasets/DIV2K_unknown | [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | **Used for the CycleSR, trim the data by referring [document](../algorithms/cyclesr.md)** |
| Cityscapes | /cache/datasets/cityscapes/ | [Download](https://www.cityscapes-dataset.com/) | **Create data index by referring [document](../algorithms/Segmentation-Adelaide-EA-NAS.md)** |
| VOC2012 | /cache/datasets/VOC2012/ | [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data) | **Create data index by referring [document](../algorithms/Segmentation-Adelaide-EA-NAS.md)** |
| ECP       | /cache/datasets/ECP/    | [Download](https://eurocity-dataset.tudelft.nl/eval/downloads/detection)  | |
| CULane | /cache/datasets/CULane/ | [Download](https://xingangpan.github.io/projects/CULane.html) | |
| Avazu | /cache/dataset/Avazu/ | [Download](https://www.kaggle.com/datasets) | |

In addition, for the following algorithm, a pre-trained model needs to be loaded. Before running the example, you need to create the directory /cache/models/, and then download the corresponding model from the corresponding location and place it in this directory:

| Algorithm | Pre-trained Model | Default Path | Model Source |
| :--: | :-- | :-- | :--: |
| Adelaide-EA | mobilenet_v2-b0353104.pth | /cache/models/mobilenet_v2-b0353104.pth | [Download](http://vega.inhuawei.com/models/pretrained/mobilenet_v2-b0353104.pth) |
| Prune-EA | resnet20.pth | /cache/models/resnet20.pth | [Download](http://vega.inhuawei.com/models/pretrained/resnet20.pth) |
| Prune-EA | resnet20.ckpt | /cache/models/resnet20.ckpt | [Download](http://vega.inhuawei.com/models/pretrained/resnet20.ckpt.tar.gz) |
| SP-NAS | resnet50-19c8e357.pth | /cache/models/resnet50-19c8e357.pth | [Download](http://vega.inhuawei.com/models/pretrained/resnet50-19c8e357.pth) |
| SP-NAS | SPNet_ECP_ImageNetPretrained_0.7978.pth | /cache/models/SPNet_ECP_ImageNetPretrained_0.7978.pth | [下载](http://vega.inhuawei.com/models/pretrained/SPNet_ECP_ImageNetPretrained_0.7978.pth) |
| SP-NAS | SPNetXB_COCO_ImageNetPretrained.pth | /cache/models/SPNetXB_COCO_ImageNetPretrained.pth | [下载](http://vega.inhuawei.com/models/pretrained/SPNetXB_COCO_ImageNetPretrained.pth) |

Note that the configuration items in the example are set to small values to speed up the running. However, if the configuration items are set to small values, the running result may be unsatisfactory. Therefore, you can modify and adjust the configuration items based on the description documents of each algorithm to obtain the required result.

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

8. AutoGate

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
    | fully train | Input | Config File: data_augmentation/cyclesr/cyclesr.yml <br> Dataset: /cache/datasets/DIV2K_unknown |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | approximate running time | n_epoch * Training time per epoch |

### 3.4 HPO

1. ASHA、BOHB、BOSS

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | hpo1 | Input | Config File: hpo/asha\|bohb\|boss/hpo/asha\|bohb\|boss.yml <br> Dataset: /cache/datasets/cifar10 |
    | hpo1 | Output | Hyperparameter file: tasks/\<task id\>/output/hpo1/best_hps.json |

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
