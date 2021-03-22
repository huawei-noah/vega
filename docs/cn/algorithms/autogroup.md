# AutoGroup

## 1. 算法介绍

AutoGroup是推荐场景的自动特征交互建模算法。推荐场景的神经网络预测模型（包括但不限于CTR预测）可简单分为三个模块：Embedding Layer，Interaction Layer以及MLP Layer，其中Interaction Layer是整个预测模型的关键模块，它需要有效的对特征交互建模。AutoGroup通过可微分学习的方式为不同阶数（order）的特征交互显式建模，并且利用新提出的交互函数保持每一阶交互的计算复杂度都与原始特征数量呈线性关系；同时AutoGroup也借鉴了随机森林（Random Forest）的思想，在每一阶特征交互建模中都使用多个特征桶融合的结果，进一步提升泛化性能。  

## 2. 算法原理

AutoGroup将有效的N阶特征交互的选择过程转化为一个结构化参数的优化问题，通过[Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf)策略将该过程可微分化。  

在训练过程中，AutoGroup交替地优化用于选择特征的结构化参数，以及模型的其他参数（如网络权重等），以达到最佳效果。

![FIS AutoGroup](../../images/fis_autogroup_overview.png)

### 2.1 搜索空间和搜索策略

搜索空间为每个feature group的feature。搜索策略上，采用Gumbel-softmax tricks 进行重参数化近似，将搜索feature group的过程转化为结构参数优化的过程，one-shot training即可搜索出最优feature group，同时，此时的模型也是最佳的模型 。

### 2.2 配置搜索空间

```yaml
fully_train:
    pipe_step:
        type: TrainPipeStep

    dataset:
        type: AvazuDataset
        common:
            data_path: /cache/datasets/avazu/

    model:
        model_desc:
            modules: ["custom"]
            custom:
	        type: AutoGroupModel
                input_dim: 645195
                input_dim4lookup: 24
                hidden_dims: [1024, 512, 256, 1]
                dropout_prob: 0.0
                batch_norm: False
                layer_norm: False
                max_order: 3
                embed_dims: [40, 60, 100]
                bucket_nums: [15, 130, 180]
                temperature: 0.01
                lambda_c: 0.01
    trainer:
        type: Trainer
        callbacks: AutoGroupTrainerCallback
        epochs: 3
        optim:
            type: Adam
            lr: 0.001
        struc_optim:
            struct_lr: 1e4
        net_optim:
            net_lr: 1e-3
        lr_scheduler:
            type: StepLR
            step_size: 1000
            gamma: 0.75
        metric:
            type: auc
        loss:
            type: BCEWithLogitsLoss

    evaluator:
        type: Evaluator
        host_evaluator:
            type: HostEvaluator
            ref: trainer
```

## 3. 使用指导

### 3.1 dataset配置

AutoGroup使用的数据是通用的CTR预测数据集格式，采用稀疏矩阵表示方式，存储为`.npy`文件。例如, 特征向量`x = [0,0,0,1,0.5,0,0,0,0,0]`可以通过这两个向量表示 `feature_id = [3, 4], feature_val = [1, 0.5]`, 第一个向量表示非零特征的id，第二个向量表示这些特征对应的取值。很多场景下`x`是二值向量，此时`feature_val`是全为1的向量，可以省略。

此处以公开数据集Avazu为例，介绍如何配置：

```yaml
dataset:
    type: AvazuDataset # data type
    common:
        data_path: /cache/datasets/avazu/ # data path
```

type为数据集名称， data_path为数据集路径
具体的数据集的配置可以参考 `./vega/core/datasets/avazu.py`

### 3.2 模型配置

AutoGroup模型需要配置基本的模型超参，也可以通过ASHA算法进行超参搜索， 下面以配置模型超参的方法为例：

```yaml
model:
    model_desc:
        modules: ["custom"]
        custom:
	    type: AutoGroupModel    # model name
            input_dim: 645195       # feature num
            input_dim4lookup: 24    # feature fields num
            hidden_dims: [1024, 512, 256, 1] # DNN part
            dropout_prob: 0.0       # dropout rate
            batch_norm: False
            layer_norm: False
            max_order: 3            # max order interaction in autogroup
            embed_dims: [40, 60, 100] # embed dimension for each order.
            bucket_nums: [15, 130, 180] # feature groups in each order
            temperature: 0.01       # gumbel-softmax parameter
            lambda_c: 0.01          # coefficient to p-th order feature self-interaction
```

### 3.3 trainer 配置

AutoGroup模型需要配置基本的trainer 超参，也可以通过ASHA算法进行超参搜索， 下面以配置trainer超参的方法为例：

```yaml
trainer:
        type: Trainer
        callbacks: AutoGroupTrainerCallback # trainer callback
        epochs: 1           # epoch num to run trainning progress
        optim:
            type: Adam      # optimism type
            lr: 0.001       # learning rate of Adam
        struc_optim:
            struct_lr: 1e4  # struct params learning rate
        net_optim:
            net_lr: 1e-3    # network params learning rate
        lr_scheduler:
            type: StepLR
            step_size: 1000
            gamma: 0.75     # learning rate decay
        metric:
            type: auc       # metrics function
        loss:
            type: BCEWithLogitsLoss # loss function, log-loss for ctr predition
```

主要需要配置struct参数的学习率和network参数的学习率。
此处与一般的算法优化不同。AutoGroup交替地优化用于选择特征的struct参数(struct optimizer)，以及模型的network参数（如网络权重等）(对应network optimizer)，以达到最佳效果。需要分别设置两个优化器的学习率。

### 3.4 算法输出

AutoGroup 算法会输出模型的best performance， 以及对应的模型权重文件。
