# 示例参考

Vega提供了算法和任务的使用指导，也针对开发者，提供了算法开发的相关指导，如扩展搜索空间和搜索算法、构建适用于Vega的数据集等。

## 1. 示例列表

在`examples`目录下如下子目录：

| 目录 | 说明 |
| :--: | :-- |
| compression | 压缩算法使用示例，包括 [Quant-EA](../algorithms/quant_ea.md)、 [Prune-EA](../algorithms/prune_ea.md) 两个算法示例 |
| data augmentation | 数据增广算法使用示例，包括 [PBA](../algorithms/pba.md)、[CycleSR](../algorithms/cyclesr.md) 算法示例 |
| hpo | 超参优化算法使用示例， 包括 [ASHA](../algorithms/hpo.md)、[BO](../algorithms/hpo.md)、[TPE](../algorithms/hpo.md)、[BOHB](../algorithms/hpo.md)、[BOSS](../algorithms/hpo.md)、Random、Random Pareto 等算法示例 |
| nas | 网络架构搜索相关示例，包括 [CARS](../algorithms/cars.md)、[SP-NAS](../algorithms/sp_nas.md)、[Auto-Lane](../algorithms/auto_lane.md)、[SR-EA](../algorithms/sr_ea.md)、[ESR-EA](../algorithms/esr_ea.md)、[Adelaide-EA](../algorithms/adelaide_ea.md)、[NAGO](../algorithms/nago.md)、BackboneNas、DartsCNN、FIS、GDAS、MFKD、SegmentationEA、SGAS、ModuleNas、DNet-NAS等算法示例 |
| fully train | fully train 相关示例，包括训练 EfficientNet B0/B4 模型示例，FMD算子示例等 |
| classification | 综合使用 NAS + HPO + FullyTrain 完成一个图像分类任务的示例 |
| features | 集群、自定义数据集、模型评估、Quota等示例 |

## 2. 运行示例

### 2.1 运行PyTorch示例

一般一个算法示例包含了是一个配置文件，有一些算法还有一些配套的代码。

进入 examples 目录后，可以执行如下命令运行示例：

```bash
vega <algorithm config file>
```

比如要运行CARS算法示例，命令如下：

```bash
vega ./nas/cars/cars.yml
```

所有的信息都在配置文件中，配置项可分为公共配置项和算法相关配置项，公共配置项可参考[配置参考](./config_reference.md)，算法配置需要参考各个算法的参考文档。

在运行示例前，需要下载数据集到缺省的数据配置目录中。在运行示例前，需要创建目录`/cache/datasets/`，然后将各个数据集下载到该目录，并解压。Vega支持Cifar10、ImageNet、COCO、Div2K、Cityscapes、VOC2012、CULane、Avazu，请用户根据需要访问各个数据集下载网址下载数据集。

另外，对于以下算法，需要加载预训练模型。在运行示例前，需要创建目录/cache/models/，然后从相应的位置下载对应的模型后，放置到该目录：

| example | Pre-trained Model | Default Path | Model Source |
| :--: | :-- | :-- | :--: |
| adelaide_ea | mobilenet_v2-b0353104.pth | /cache/models/mobilenet_v2-b0353104.pth | [download](https://box.saas.huaweicloud.com/p/e9e06f49505a1959da6cba9401b2bf38) |
| BackboneNas (mindspore) | resnet50-19c8e357.pth | /cache/models/resnet50-19c8e357.pth | [download](https://box.saas.huaweicloud.com/p/f2ab3a1869f55de2053fb1404fc1c6d3) |
| BackboneNas (tensorflow), classification, prune_ea(tensorflow) | resnet_imagenet_v1_fp32_20181001 | /cache/models/resnet_imagenet_v1_fp32_20181001/ <br>  keep only these files: checkpoint, graph.pbtxt, model.ckpt-225207.data-00000-of-00002, model.ckpt-225207.data-00001-of-00002, model.ckpt-225207.index, model.ckpt-225207.meta | [download](http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v1_fp32_20181001.tar.gz) |
| dnet_nas | 031-_64_12-1111-11211112-2.pth | /cache/models/031-_64_12-1111-11211112-2.pth | [download](https://box.saas.huaweicloud.com/p/3e5b678ec970ad347c678fabde23fb27) |
| prune_ea(pytorch) | resnet20.pth | /cache/models/resnet20.pth | [download](https://box.saas.huaweicloud.com/p/67cd96e5da41b1c5a88f2b323446c0f8) |
| prune_ea(mindspore) | resnet20.ckpt | /cache/models/resnet20.ckpt | [download](https://box.saas.huaweicloud.com/p/7f1743a041a0ede7f68713d1360a57d5) |
| sp_nas | fasterrcnn_coco.pth | /cache/models/fasterrcnn_coco.pth | [download](https://box.saas.huaweicloud.com/p/27882e18e1594eaba79ff4a827e136d7) |
| sp_nas | fasterrcnn_serialnet_backbone.pth | /cache/models/fasterrcnn_serialnet_backbone.pth | [download](https://box.saas.huaweicloud.com/p/60d2fb4d65533b60f336e74aaaeb5d96) |
| sp_nas | serial_classification_net.pth | /cache/models/serial_classification_net.pth | [download](https://box.saas.huaweicloud.com/p/9a22840425a6ed1b53f873c506488c1c) |
| sp_nas | torch_fpn.pth | /cache/models/torch_fpn.pth | [download](https://box.saas.huaweicloud.com/p/0bb0b0cf5229368fac006b1d8955df5b) |
| sp_nas | torch_rpn.pth | /cache/models/torch_rpn.pth | [download](https://box.saas.huaweicloud.com/p/a41b7ca75114a9b2488e4064ed6ba3fb) |

在每个示例的配置文件中，在 `general/backend` 中有该示例适用的平台说明（PyTorch、TensorFlow、MindSpore）。

比如以下配置说明该示例可以运行在三个平台中：

```yaml
general:
    backend: pytorch  # pytorch | tensorflow | mindspore
```

以下配置只能运行在TensorFlow：

```yaml
general:
    backend: tensorflow
```

### 2.2 运行TensorFlow示例

1. 运行命令（GPU）：

    ```bash
    vega <algorithm config file> -b t
    ```

    如：

    ```bash
    vega ./nas/backbone_nas/backbone_nas.yml -b t
    ```

2. 运行命令（Ascend 910）：

    ```bash
    vega <algorithm config file> -b t -d NPU
    ```

    如：

    ```bash
    vega ./nas/backbone_nas/backbone_nas.yml -b t -d NPU
    ```

### 2.3 运行MindSpore示例

运行命令（Ascend 910）：

```bash
vega <algorithm config file> -b m -d NPU
```

如：

```bash
vega ./nas/backbone_nas/backbone_nas.yml -b m -d NPU
```

## 3. 示例输入和输出

### 3.1 模型压缩示例

1. Prune-EA

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | nas | 输入 | 配置文件：compression/prune-ea/prune.yml <br> 预训练模型：/cache/models/resnet20.pth <br> 数据集：/cache/datasets/cifar10 |
    | nas | 输出 | 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | nas | 运行时间估算 | (random_samples + num_generation * num_individual) * epochs / GPU数 * 1个epoch的训练时间 |
    | fully train | 输入 | 配置文件：compression/prune-ea/prune.yml <br> 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> 数据集：/cache/datasets/cifar10 |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

2. Quant-EA

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | nas | 输入 | 配置文件：compression/quant-ea/quant.yml <br> 数据集：/cache/datasets/cifar10 |
    | nas | 输出 | 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | nas | 运行时间估算 | (random_samples + num_generation * num_individual) * epochs / GPU数 * 1个epoch的训练时间 |
    | fully train | 输入 | 配置文件：compression/quant-ea/quant.yml <br> 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> 数据集：/cache/datasets/cifar10 |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

### 3.2 网络架构搜索

1. CARS

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | nas | 输入 | 配置文件：nas/cars/cars.yml <br> 数据集：/cache/datasets/cifar10 |
    | nas | 输出 | 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json |
    | nas | 运行时间估算 | epochs * 1个epoch的训练时间 (epoch的时间受到num_individual的影响) |
    | fully train | 输入 | 配置文件：nas/cars/cars.yml <br> 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json <br> 数据集：/cache/datasets/cifar10 |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

2. Adelaide-EA

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | random | 输入 | 配置文件：nas/adelaide_ea/adelaide_ea.yml <br> 数据集：/cache/datasets/cityscapes |
    | random | 输出 | 网络描述文件：tasks/\<task id\>/output/random/model_desc_\<id\>.json |
    | random | 运行时间估算 | max_sample * epochs / GPU数目 * 1个epoch的训练时间 |
    | mutate | 输入 | 配置文件：nas/adelaide_ea/adelaide_ea.yml <br> 数据集：/cache/datasets/cityscapes <br> 网络描述文件：tasks/\<task id\>/output/random/model_desc_\<id\>.json  |
    | mutate | 输出 | 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json |
    | mutate | 运行时间估算 | max_sample * epochs / GPU数目 * 1个epoch的训练时间 |
    | fully train | 输入 | 配置文件：nas/adelaide_ea/adelaide_ea.yml <br> 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> 数据集：/cache/datasets/cityscapes |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

3. ESR-EA

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | nas | 输入 | 配置文件：nas/esr_ea/esr_ea.yml <br> 数据集：/cache/datasets/DIV2K |
    | nas | 输出 | 网络描述文件：tasks/\<task id\>/output/nas/selected_arch.npy |
    | nas | 运行时间估算 |  num_generation * num_individual * epochs / GPU数目 * 1个epoch的训练时间 |
    | fully train | 输入 | 配置文件：nas/esr_ea/esr_ea.yml <br> 网络描述文件：tasks/\<task id\>/output/nas/selected_arch.npy <br> 数据集：/cache/datasets/DIV2K |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

4. SR-EA

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | random | 输入 | 配置文件：nas/sr_ea/sr_ea.yml <br> 数据集：/cache/datasets/DIV2K |
    | random | 输出 | 网络描述文件：tasks/\<task id\>/output/random/model_desc_\<id\>.json |
    | random | 运行时间估算 | num_sample * epochs / GPU数目 * 1个epoch的训练时间 |
    | mutate | 输入 | 配置文件：nas/sr_ea/sr_ea.yml <br> 数据集：/cache/datasets/DIV2K <br> 网络描述文件：tasks/\<task id\>/output/random/model_desc_\<id\>.json  |
    | mutate | 输出 | 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json |
    | mutate | 运行时间估算 | num_sample * num_mutate * epochs / GPU数目 * 1个epoch的训练时间 |
    | fully train | 输入 | 配置文件：nas/sr_ea/sr_ea.yml <br> 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> 数据集：/cache/datasets/DIV2K |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

5. SP-NAS

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | nas1 | 输入 | 配置文件：nas/sp_nas/spnas.yml <br> 数据集：/cache/datasets/COCO2017 <br> 预训练模型：/cache/models/resnet50-19c8e357.pth <br> 配置文件： nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | nas1 | 输出 | 网络描述文件：tasks/\<task id\>/output/nas1/model_desc_\<id\>.json <br> 模型列表：tasks/\<task id\>/output/total_list_p.csv |
    | nas1 | 运行时间估算 | max_sample * epoch / GPU数目 * 1个epoch的训练时间 |
    | nas2 | 输入 | 配置文件：nas/sp_nas/spnas.yml <br> 数据集：/cache/datasets/COCO2017 <br> 网络描述文件：tasks/\<task id\>/output/nas1/model_desc_\<id\>.json <br> 模型列表：tasks/\<task id\>/output/total_list_p.csv <br> 配置文件： nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | nas2 | 输出 | 网络描述文件：tasks/\<task id\>/output/nas2/model_desc_\<id\>.json <br> 模型列表：tasks/\<task id\>/output/total_list_s.csv |
    | nas2 | 运行时间估算 | max_sample * epoch / GPU数目 * 1个epoch的训练时间 |
    | fully train | 输入 |  配置文件：nas/sp_nas/spnas.yml <br> 数据集：/cache/datasets/COCO2017 <br> 网络描述文件：tasks/\<task id\>/output/nas2/model_desc_\<id\>.json <br> 模型列表：tasks/\<task id\>/output/total_list_s.csv <br> 配置文件： nas/sp_nas/faster_rcnn_r50_fpn_1x.py |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fullytrain/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

6. Auto_Lane

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    |nas|输入|配置文件：nas/auto_lane/auto_lane.yml <br/>数据集：/cache/datasets/CULane  OR  /cache/datasets/CurveLane|
    |nas|输出|网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json|
    |nas|运行时间估算|max_sample * epoch / GPU数目 * 1个epoch的训练时间|
    |fully train|输入|配置文件：nas/sp_nas/auto_lane.yml <br> 数据集：/cache/datasets/CULane  OR      /cache/datasets/CurveLane<br> 网络描述文件：tasks/\<task id\>/output/nas/model_desc_\<id\>.json|
    |fully train|输出|模型：tasks/\<task id\>/output/fullytrain/model_\<id\>.pth|
    |fully train|运行时间估算|epochs * 1个epoch的训练时间|

7. AutoGroup

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | fully train | 输入 | 配置文件 nas/fis/autogroup.yml <br> 数据集"/cache/datasets/avazu |
    | fully train | 输出 | 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> 模型：tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

8. AutoFis

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | search | 输入 | 配置文件 nas/fis/autogate_grda.yml <br> 数据集"/cache/datasets/avazu |
    | search | 输出 | 模型：tasks/\<task id\>/output/search/0/model.pth |
    | search | 运行时间估算 | epochs * 1个epoch的训练时间 |
    | retrain | 输入 | 配置文件 nas/fis/autogate_grda.yml <br> 数据集"/cache/datasets/avazu |
    | retrain | 输出 | 模型：tasks/\<task id\>/output/retrain/0/model.pth |
    | retrain | 运行时间估算 | epochs * 1个epoch的训练时间 |

### 3.3 数据增广

1. PBA

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | pba | 输入 | 配置文件：data_augmentation/pba/pba.yml <br> 数据集：/cache/datasets/cifar10 |
    | pba | 输出 | Transformer列表：tasks/\<task id\>/output/pba/best_hps.json |
    | pba | 运行时间估算 | total_rungs * each_epochs * config_count / GPU数目 * 1个epoch的训练时间 |

2. CycleSR

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | fully train | 输入 | 配置文件：data_augmentation/cyclesr/cyclesr.yml <br> 数据集：/cache/datasets/DIV2K_unpair |
    | fully train | 输出 | 模型：tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | 运行时间估算 |  n_epoch * 1个epoch的训练时间 |

### 3.4 超参优化

1. ASHA、BOHB、BOSS

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | hpo | 输入 | 配置文件：hpo/asha\|bohb\|boss/hpo/asha\|bohb\|boss.yml <br> 数据集：/cache/datasets/cifar10 |
    | hpo | 输出 | 超参描述文件：tasks/\<task id\>/output/hpo/best_hps.json |

### 3.5 Fully Train

1. EfficientNet

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | fully train | 输入 | 配置文件：fully_train/efficientnet/efficientnet_b0.yml <br> 数据集：/cache/datasets/ILSVRC |
    | fully train | 输出 | 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> 模型：tasks/\<task id\>/output/fully_train/model_\<id\>.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

2. FMD

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | fully train | 输入 | 配置文件：fully_train/fmd/fmd.yml <br> 数据集：/cache/datasets/cifar10 |
    | fully train | 输出 | 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> 模型：tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |

3. ResNet

    | 阶段 | 选项 | 内容 |
    | :--: | :--: | :-- |
    | fully train | 输入 | 配置文件 fully_train/trainer/resnet.yml <br> 数据集"/cache/datasets/ILSVRC |
    | fully train | 输出 | 网络描述文件：tasks/\<task id\>/output/mutate/model_desc_\<id\>.json <br> 模型：tasks/\<task id\>/output/fully_train/model_0.pth |
    | fully train | 运行时间估算 | epochs * 1个epoch的训练时间 |
