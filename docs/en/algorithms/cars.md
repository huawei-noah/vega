# CARS: Continuous Evolution for Efficient Neural Architecture Search

## 1. Algorithm Introduction

Many existing NAS methods can only get one network structure during the search iteration, and cannot take into account various hardware constraints, like power, performance, latency etc. In addition, although the evolutionary algorithm-based NAS method achieves good performance, each generation of samples needs to be retrained for evaluation, which greatly affects the search efficiency. Considering the insufficiency of the existing NAS methods, we propose a continuous evolution approach called Efficient Neural Architecture Search (CARS). CARS maintains an optimal model solution set, and updates the parameters of the supernet through the model in the solution set. In each iteration of evolutionary algorithm, current samples can directly inherit the parameters from the supernet and parent samples, which effectively improves the efficiency of evolution. CARS can obtain a series of models with different sizes and precisions in one search. Users can select the corresponding model based on the resource constraints in the actual application. For details about the CARS algorithm, see  HYPERLINK "https://arxiv.org/abs/1909.04977" https://arxiv.org/abs/1909.04977. 

## 2. Algorithm Principles

The following figure shows the principle of the CARS algorithm. The CARS algorithm maintains a supernet, and each subnet is a sample of the supernet. In the process of evolutionary algorithm, the weight of all the subnets is shared with the supernet, so that the samples of the sub-generation can inherit the weight of the samples of the parent generation directly, thus realizing continuous evolution. CARS algorithm uses pNSGA - III algorithm to protect large models in the searching process and increase the coverage of the searching model. The CARS algorithm uses the subnets in the Pareto frontier to update the weight of supernet. The weight update of supernet and the evolutionary update of subnets are performed alternately.

![framework](../../images/cars_framework.png)

### 2.1 Search Space

#### DARTS search space

The CARS in the pipeline integrates the search space of DARTS. The overall structure is as follows:

![darts_search_sapce](../../images/cars_darts_search_sapce.png)

For a detailed description of the DARTS search space, see the corresponding  HYPERLINK "https://arxiv.org/abs/1806.09055" ICLR '19 article.

### 2.2 Configuring the Search Space

The search space configuration is as follows:

```yaml
    search_space:
        type: SearchSpace
        modules: ['super_network']
        super_network:
           type: CARSDartsNetwork
           stem:
               type: PreOneStem
                init_channels: 16
                stem_multi: 3
            head:
                type: LinearClassificationHead
            init_channels: 16
            num_classes: 10
            auxiliary: False
            search: True
            cells:
                modules: [
                    'normal', 'normal', 'reduce',
                    'normal', 'normal', 'reduce',
                    'normal', 'normal'
                ]
                normal:
                    type: NormalCell
                    steps: 4
                    genotype:
                      [
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 4 ],
                      ]
                    concat: [2, 3, 4, 5]
                reduce:
                    type: ReduceCell
                    steps: 4
                    genotype:
                      [
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 2, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 3, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 4, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 0 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 1 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 2 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 3 ],
                      [ ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'], 5, 4 ],
                      ]
                    concat: [2, 3, 4, 5]
```

## 3. Usage Guide

### 3.1 Dataset Configuration

By default, the CARS uses the CIFAR-10 data set. You can also use customized data sets. To use datasets in user-defined format, you need to implement dataset interface that meet Vega requirements. For details, see the development manual.

The configuration information of the CIFAR-10 database is as follows:

```yaml
    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
            num_workers: 8
            train_portion: 0.5
            drop_last: False
        train:
            shuffle: True
            batch_size: 128
        val:
            batch_size: 3500
```

### 3.2 Running Environment Settings

For details about how to search and train a model, see the following configuration file for parameter setting:

- vega/examples/nas/cars/cars.yml

The configuration file is directly transferred to the pipeline through main.py. The two processes are performed in sequence. During the search process, a series of models at the Pareto Front are found. During the training process, the selected models are fully trained to obtain the final performance.

The configuration of the search process is as follows:

```yaml
nas:
    pipe_step:
        type: SearchPipeStep

    search_algorithm:
        type: CARSAlgorithm
        policy:
            num_individual: 128
            start_ga_epoch: 50
            ga_interval: 10
            select_method: uniform #pareto
            warmup: 50

    trainer:
        type: Trainer
        darts_template_file: "{default_darts_cifar10_template}"
        callbacks: CARSTrainerCallback
        model_statistics: False
        epochs: 500
        optim:
            type: SGD
	    params:
	        lr: 0.025
	        momentum: 0.9
	        weight_decay: !!float 3e-4
        lr_scheduler:
            type: CosineAnnealingLR
	        params:
                T_max: 500
                eta_min: 0.001
        loss:
            type: CrossEntropyLoss

        grad_clip: 5.0
        seed: 10
        unrolled: True
```

Configuration of the fully train phase:

```yaml
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/nas/"

    trainer:
        ref: nas.trainer
        callbacks: DartsFullTrainerCallback
        epochs: 600
        lr_scheduler:
            type: CosineAnnealingLR
	        params:
                T_max: 600.0
                eta_min: 0
        loss:
            type: MixAuxiliaryLoss
	        params:
                loss_base:
                    type: CrossEntropyLoss

            aux_weight: 0.4
        seed: 100
        drop_path_prob: 0.2

    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
            num_workers: 8
            drop_last: False
            batch_size: 96
        train:
            shuffle: True
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
                - type: Cutout
                  length: 8
        val:
            batch_size: 96
            shuffle: False
```

The models_folder and model_desc_n.json parameters in the fully train are described as follows:

1. models_folder: specifies a directory where one or more model_desc_n.json files are stored. n indicates the sequence number. This parameter is mutually exclusive with model_desc_file. If both of them are set, model_desc_file are preferred.
2. model_desc_file: specifies a model description file as the input of the trainer.

### 3.3 Algorithm Output

The output:

1. nas directory: outputs multiple optimal model description files.
2. fully_train: outputs the trained model weight file.

## 4. Benchmark

For details, see the benchmark configuration item in the [cars.yml](https://github.com/huawei-noah/vega/blob/master/examples/nas/cars/cars.yml) configuration file.
