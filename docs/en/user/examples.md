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

| Dataset | Default Path | Data Source | 
| :--- | :--- | :--: | 
| Cifar10 | /cache/datasets/cifar10/ | [Download](https://www.cs.toronto.edu/~kriz/cifar.html) |
| ImageNet | /cache/datasets/ILSVRC/ | [Download](http://image-net.org/download-images) |
| COCO | /cache/datasets/COCO2017 | [Download](http://cocodataset.org/#download) |
| Div2K | /cache/datasets/DIV2K/ | [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Div2kUnpair | /cache/datasets/DIV2K_unknown | [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Cityscapes | /cache/datasets/cityscapes/ | [Download](https://www.cityscapes-dataset.com/) |
| VOC2012 | /cache/datasets/VOC2012/ | [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data) |
| Cifar10TF | /cache/datasets/cifar-10-batches-bin/ | [Download](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) |
| ECP       | /cache/datasets/ECP/    | [Download](https://eurocity-dataset.tudelft.nl/eval/downloads/detection)  |

In addition, for the following algorithm, a pre-trained model needs to be loaded. Before running the example, you need to create the directory /cache/models/, and then download the corresponding model from the corresponding location and place it in this directory:

| Algorithm | Pre-trained Model | Default Path | Model Source |
| :--: | :-- | :-- | :--: |
| Adelaide-EA | mobilenet_v2-b0353104.pth | /cache/models/mobilenet_v2-b0353104.pth | [Download](http://www.noahlab.com.hk/opensource/vega/models/pretrained/mobilenet_v2-b0353104.pth) |
| Prune-EA | resnet20.pth | /cache/models/resnet20.pth | [Download](http://www.noahlab.com.hk/opensource/vega/models/pretrained/resnet20.pth) |
| SP-NAS | resnet50-19c8e357.pth | /cache/models/resnet50-19c8e357.pth | [Download](http://www.noahlab.com.hk/opensource/vega/models/pretrained/resnet50-19c8e357.pth) |
|        | SPNet_ECP_ImageNetPretrained_0.7978.pth | /cache/models/SPNet_ECP_ImageNetPretrained_0.7978.pth | [Download](http://www.noahlab.com.hk/opensource/vega/models/pretrained/SPNet_ECP_ImageNetPretrained_0.7978.pth) |

Note that the configuration items in the example are set to small values to speed up the running. However, if the configuration items are set to small values, the running result may be unsatisfactory. Therefore, you can modify and adjust the configuration items based on the description documents of each algorithm to obtain the required result.

## 3. Examples' Input and Output

### 3.1 Model Compression

1. Prune-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: compression/prune-ea/prune.yml <br> Pre-Trained Model: /cache/models/resnet20.pth <br> Dataset: /cache/datasets/cifar10 |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | fully train | Input | Config File: compression/prune-ea/prune.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |

2. Quant-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: compression/quant-ea/quant.yml <br> Dataset: /cache/datasets/cifar10 |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | fully train | Input | Config File: compression/quant-ea/quant.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |

### 3.2 NAS

1. CARS

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: nas/cars/cars.yml <br> Dataset: /cache/datasets/cifar10 |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | fully train | Input | Config File: nas/cars/cars.yml <br> Network Description File: tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |

2. Adelaide-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | random | Input | Config File: nas/adelaide_ea/adelaide_ea.yml <br> Dataset: /cache/datasets/cityscapes |
    | random | Output | Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json |
    | mutate | Input | Config File: nas/adelaide_ea/adelaide_ea.yml <br> Dataset: /cache/datasets/cityscapes <br> Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json  |
    | mutate | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json |
    | fully train | Input | Config File: nas/adelaide_ea/adelaide_ea.yml <br> Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Dataset: /cache/datasets/cityscapes |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |

3. ESR-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas | Input | Config File: nas/esr_ea/esr_ea.yml <br> Dataset: /cache/datasets/DIV2K |
    | nas | Output | Network Description File: tasks/\<task id\>/output/nas/selected_arch.npy |
    | fully train | Input | Config File: nas/esr_ea/esr_ea.yml <br> Network Description File: tasks/\<task id\>/output/nas/selected_arch.npy <br> Dataset: /cache/datasets/DIV2K |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |

4. SR-EA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | random | Input | Config File: nas/sr_ea/sr_ea.yml <br> Dataset: /cache/datasets/DIV2K |
    | random | Output | Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json |
    | mutate | Input | Config File: nas/sr_ea/sr_ea.yml <br> Dataset: /cache/datasets/DIV2K <br> Network Description File: tasks/\<task id\>/output/random/model_desc_\<id\>.json  |
    | mutate | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json |
    | fully train | Input | Config File: nas/sr_ea/sr_ea.yml <br> Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Dataset: /cache/datasets/DIV2K |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_\<id\>.pth |

5. SP-NAS

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | nas1 | Input | Config File: nas/sp_nas/spnas.yml <br> Dataset: /cache/datasets/COCO2017 <br> Pre-Trained Model: /cache/models/resnet50-19c8e357.pth <br> Config File:  nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | nas1 | Output | Network Description File: tasks/\<task id\>/output/nas1/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/nas1/total_list_p.csv |
    | nas2 | Input | Config File: nas/sp_nas/spnas.yml <br> Dataset: /cache/datasets/COCO2017 <br> Network Description File: tasks/\<task id\>/output/nas1/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/nas1/total_list_p.csv <br> Config File:  nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | nas2 | Output | Network Description File: tasks/\<task id\>/output/nas2/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/nas2/total_list_s.csv |
    | fully train | Input |  Config File: nas/sp_nas/spnas.yml <br> Dataset: /cache/datasets/COCO2017 <br> Network Description File: tasks/\<task id\>/output/nas2/model_desc_\<id\>.json <br> Model List: tasks/\<task id\>/output/nas2/total_list_s.csv <br> Config File:  nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | fully train | Output | Model: tasks/\<task id\>/output/fullytrain/model_\<id\>.pth |

### 3.3 Data Augmentation

1. PBA

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | pba | Input | Config File: data_augmentation/pba/pba.yml <br> Dataset: /cache/datasets/cifar10 |
    | pba | Output | Transformer List: tasks/\<task id\>/output/pba/best_hps.json |

2. CycleSR

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File: data_augmentation/cyclesr/cyclesr.yml <br> Dataset: /cache/datasets/DIV2K_unknown |
    | fully train | Output | Model: tasks/\<task id\>/output/fully_train/model_0.pth |

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

2. FMD

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File: fully_train/fmd/fmd.yml <br> Dataset: /cache/datasets/cifar10 |
    | fully train | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Model: tasks/\<task id\>/output/fully_train/model_0.pth |

3. FMD

    | Stage | Option | Content |
    | :--: | :--: | :-- |
    | fully train | Input | Config File fully_train/trainer/resnet.yml <br> Dataset: /cache/datasets/ILSVRC |
    | fully train | Output | Network Description File: tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> Model: tasks/\<task id\>/output/fully_train/model_0.pth |
