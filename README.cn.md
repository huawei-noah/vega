# Vega

[English](./README.md)

**Vega ver1.0.0 发布：**

- 新增算法：[auto-lane](./docs/cn/algorithms/auto_lane.md)，[AutoFIS](./docs/cn/algorithms/fis-autogate.md)，[AutoGroup](./docs/cn/algorithms/fis-autogroup.md)，[MFKD](https://arxiv.org/pdf/2006.08341.pdf)。
- 特性增强：
  - Trainer提供回调机制：Trainer支持Callback机制，并提供九个缺省的callback。
  - Report机制：提供统一的AutoML算法的数据收集和处理机制。
  - 多Backend：提供TensorFlow支持，欢迎试用。
  - 评估服务：提供独立的评估服务，提供Atalas 200DK和Bolt的模型评估。
- 社区贡献者：[qixiuai](https://github.com/qixiuai), [hasanirtiza](https://github.com/hasanirtiza), [cndylan](https://github.com/cndylan)，[IlyaTrofimov](https://github.com/IlyaTrofimov)。


## Vega简介

Vega是诺亚方舟实验室自研的AutoML算法工具链，有主要特点：

1. 完备的AutoML能力：涵盖HPO(超参优化, HyperParameter Optimization)、Data-Augmentation、NAS(网络架构搜索， Network Architecture Search)、Model Compression、Fully Train等关键功能，同时这些功能自身都是高度解耦的，可以根据需要进行配置，构造完整的pipeline。
2. 业界标杆的自研算法：提供了诺亚方舟实验室自研的 **[业界标杆](./docs/cn/benchmark/benchmark.md)** 算法，并提供 **[Model Zoo](./docs/cn/model_zoo/model_zoo.md)** 下载SOTA(State-of-the-art)模型。
3. 高并发模型训练能力：提供高性能Trainer，加速模型训练和评估。
4. 多Backend支持：支持PyTorch，TensorFlow（试用中），MindSpore（开发中）。

## 算法列表

| 分类 | 算法 | 说明 | 参考 |
| :--: | :-- | :-- | :-- |
| NAS | [SM-NAS: Structural-to-Modular NAS](https://arxiv.org/abs/1911.09929) | 两阶段物体检测架构搜索算法 | 开发中 |
| NAS | [CARS: Continuous Evolution for Efficient Neural Architecture Search](https://arxiv.org/abs/1909.04977) | 基于连续进化的多目标高效神经网络结构搜索方法 | [参考](./docs/cn/algorithms/cars.md) |
| NAS | SR-EA | 适用于轻量级网络的自动网络架构搜索方法 | [参考](./docs/cn/algorithms/sr-ea.md) |
| NAS | [ESR-EA: Efficient Residual Dense Block Search for Image Super-resolution](https://arxiv.org/abs/1909.11409) | 基于网络架构搜索的多目标图像超分方法 | [参考](./docs/cn/algorithms/esr_ea.md) |
| NAS | [Adelaide-EA: SEGMENTATION-Adelaide-EA-NAS](https://arxiv.org/abs/1810.10804) | 图像分割网络架构搜索算法 | [参考](./docs/cn/algorithms/Segmentation-Adelaide-EA-NAS.md) |
| NAS | [SP-NAS: Serial-to-Parallel Backbone Search for Object Detection](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_SP-NAS_Serial-to-Parallel_Backbone_Search_for_Object_Detection_CVPR_2020_paper.pdf) | 面向物体检测及语义分割的高效主干网络架构搜索算法 | [参考](./docs/cn/algorithms/sp-nas.md) |
| NAS | Auto-Lane: CurveLane-NAS | 一种端到端的车道线架构搜索算法 | [参考](./docs/cn/algorithms/auto_lane.md) |
| NAS | [AutoFIS](https://arxiv.org/pdf/2003.11235.pdf) | 一种适用于推荐搜索场景下的自动特征选择算法 | [参考](./docs/cn/algorithms/fis-autogate.md) |
| NAS | [AutoGroup](https://dl.acm.org/doi/pdf/10.1145/3397271.3401082) | 一种适用于推荐搜素场景下的自动特征交互建模算法 | [参考](./docs/cn/algorithms/fis-autogroup.md) |
| Model Compression | Quant-EA: Quantization based on Evolutionary Algorithm | 自动混合比特量化算法，使用进化策略对CNN网络结构每层量化位宽进行搜索 | [参考](./docs/cn/algorithms/quant_ea.md) |
| Model Compression | Prune-EA | 使用进化策略对CNN网络结构进行自动剪枝压缩算法 | [参考](./docs/cn/algorithms/prune_ea.md) |
| HPO | [ASHA: Asynchronous Successive Halving Algorithm](https://arxiv.org/abs/1810.05934) | 动态连续减半算法 | [参考](./docs/cn/algorithms/hpo.md) |
| HPO | [TPE: Tree-structured Parzen Estimator Approach](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) | 一种基于树状结构Parzen估计方法的超参优化算法  | [参考](./docs/cn/algorithms/hpo.md) |
| HPO | BO: Bayesian Optimization | 贝叶斯优化算法 | [参考](./docs/cn/algorithms/hpo.md) |
| HPO | [BOHB: Hyperband with Bayesian Optimization](https://arxiv.org/abs/1807.01774) | 动态连续减半算法 | [参考](./docs/cn/algorithms/hpo.md) |
| HPO | BOSS: Bayesian Optimization via Sub-Sampling | 基于贝叶斯优化框架下的一种针对计算资源受限，需要高效搜索的，具有普适性的超参优化算法 | [参考](./docs/cn/algorithms/hpo.md) |
| Data Augmentation | [PBA: Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules](https://arxiv.org/abs/1905.05393) | 基于PBT优化算法搜索数据增广策略时刻表的数据增广算法 | [参考](./docs/cn/algorithms/pba.md) |
| Data Augmentation | cyclesr: CycleGAN + SR | 底层视觉的无监督风格迁移算法 | [参考](./docs/cn/algorithms/cyclesr.md) |
| Fully Train | [FMD](https://arxiv.org/abs/2002.11022) | 基于特征图扰动的神经网络训练方法 | [参考](./docs/cn/algorithms/fmd.md) |

## 获取和安装

请获取最新版本，参考 **[安装指导](./docs/cn/user/install.md)** 完成安装，若希望在集群中部署Vega，请参考 **[部署指导](./docs/cn/user/deployment.md)** 。

## 使用指导

Vega高度模块化，可通过配置可完成搜索空间、搜索算法、pipeline的构建，运行Vega应用就是加载配置文件，并根据配置来完成AutoML流程。
同时Vega提供了详细的操作示例供大家参考，可参考 **[示例参考](./docs/cn/user/examples.md)** ，如运行CARS算法示例：

```bash
cd examples
python3 ./run_example.py ./nas/cars/cars.yml pytorch
```

在使用Vega前，有必要充分了解配置项的含义，请参考 **[配置指导](./docs/cn/user/config_reference.md)** 。

**注意：**

在运行示例前，需要在算法配置文件中配置数据集和预训练模型目录，缺省目录请参考 **[示例参考](./docs/cn/user/examples.md)** 。

## 开发者指导

Vega框架组件解耦，并采用注册机制来组合各个功能组件，便于扩充功能和算法，可参考 **[开发者指导](./docs/cn/developer/developer_guide.md)** ，了解Vega架构和主要机制。

同时可参考 **[快速入门指导](./docs/cn/developer/quick_start.md)** ，通过实现一个简单的CNN网络搜索功能，通过实战快速进入Vega应用开发。

在Vega应用的开发中，最先遇到的问题就是，如何引入业务数据集到Vega中，可参考 **[数据集指导](./docs/cn/developer/datasets.md)** 。

针对不同的算法，可参考 **[算法开发指导](./docs/cn/developer/new_algorithm.md)** ，可根据文中提供的示例，一步步的将新算法添加到Vega中。

在Automl的大多数算法中搜索空间和网络是强相关强耦合的，我们尝试统一搜索空间的定义方式，使得同一种搜索空间能够适配不同的搜索算法，我们称之为 **[细粒度搜索空间指导](./docs/cn/developer/fine_grained_search_space.md)** ，欢迎大家尝试使用。

当然文档解决不了所有的疑问和问题，若你在使用中遇到任何问题，请及时通过issue反馈，我们会及时答复和解决。

## 参考列表

| 对象 | 参考 |
| :--: | :-- |
| 用户 | [安装指导](./docs/cn/user/install.md)、[部署指导](./docs/cn/user/deployment.md)、[配置指导](./docs/cn/user/config_reference.md)、[示例参考](./docs/cn/user/examples.md)、[评估服务](../docs/cn/user/evaluate_service.md)、任务参考([分类](./docs/cn/tasks/classification.md)、[检测](./docs/cn/tasks/detection.md)、[分割](./docs/cn/tasks/segmentation.md)、[超分](./docs/cn/tasks/segmentation.md)) |
| 开发者 | [开发者指导](./docs/cn/developer/developer_guide.md)、[快速入门指导](./docs/cn/developer/quick_start.md)、[数据集指导](./docs/cn/developer/datasets.md)、[算法开发指导](./docs/cn/developer/new_algorithm.md)、[细粒度搜索空间指导](./docs/cn/developer/fine_grained_search_space.md) |

## 合作和贡献

欢迎大家使用Vega，有任何疑问、求助、修改bug、贡献算法、完善文档，请在社区提交issue，我们会及时回复沟通交流。
欢迎大家加入我们的QQ群: **833345709** 。
