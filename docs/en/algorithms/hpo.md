# HPO

## 1. Introduction to hyperparameter optimization Functions

### 1.1 Algorithms

The hyperparameter optimization functions provided in vega.algorithms.hpo module: 

1. Single-objective hyperparameter optimization 

- [x] Random
- [x] BO (Note: Currently supports SMAC and GPEI.)
- [x] ASHA
- [x] BOHB (Note: Asynchronous parallelism is not included in the current implementation. The BO model is GP+EI.)
- [x] BOSS (Note: Asynchronous parallelism is not included in the current implementation. The BO model is GP+EI.)
- [x] TPE

2. Multi-objective hyperparameter optimization

- [x] RandomPareto

### 1.2 Hyperparameter Search Space

Vega provides a genernalized search space that can be composed of independent discrete or continuous variables, and allows users to set the conditional constraint relationship between variables. The search space is defined in  vega.algorithms.hpo.hpyperparameter_space. Current search space supports the following variable types and constraints between variables:

1. **Variable Type**

Continuous variables:

- [x] INT
- [x] INT_EXP
- [x] FLOAT
- [x] FLOAT_EXP

Discrete variables:

- [x] INT_CAT
- [x] FLOAT_CAT
- [x] STRING
- [x] BOOL

2. **Condition Constraint Type**

- [x] EQUAL
- [x] NOT_EQUAL
- [x] IN

A general design of the search space can be find in Chapter 3.

### 1.3 HPO Module

Vega provides an general pipeline and generators for multiple hyperparameter optimization algorithms. Users can assign the hyperparameter search space and hyperparameter search algorithm in the configuration file, provide the related trainer and evaluator. By invoking the hpo_pipestep, users can use the HPO function conveniently. For details, see chapter 4.

## 2. Introduction to Hyperparameter Optimization Algorithms

### 2.1 Hyperparameter Optimization for Deep Neural Network

**Widely used hyperparameter optimization methods:**

1. Grid search
2. Random search
3. Bayesian optimization
4. Dynamic Resource Allocation

### 2.2 Dynamic Resource Allocation

Stochastic gradient descent is a well-known optimization method for neural network training. A batch of learning curves and validation performance can be used to evaluate whether the hyperparameter candidates are good. If the searched hyperparameters are not optimal, we can apply an early-stopping strategy to abort the current training.

**Advantages:**

- Different from traditional random search or bayesian optimization,  dynamic resource allocation method is more suitable for neural network training, which requires multiple iterations, and a huge amount of computing resources. Dynamic resource allocation can find better hyperparameter combinations fast with less computating resource consumption, multiple GPUs can be used to asynchronously search for the optimal hyperparameter combination in parallel.

**Dynamic resource allocation based HPO with early stop policy are as follows:**

- Successive Halving Algorithm (SHA).
- Asynchronous Successive Halving Algorithm (ASHA).
- Bandit-Based Hyperparameter Optimization (HyperBand).
- Hyperband with Bayesian Optimization (BOHB).
- Population Based Training (PBT).

## 3. Introduction to HyperparameterSpace

Currently, Vega Pipeline provides a general search space that can contains independent discrete or continuous variables, and allows to set the conditional constraint relationship between variables (vega.algorithms.hpo.hyperparameter_space). This chapter describes the architecture design of the Hyperparameter Space. For details about how to use the HyperparameterSpace, see chapter "."

Overall architecture of HyperparameterSpace:

![sha.png](../../images/hyperparameter_space_1.png)

### 3.1 HyperparameterSpace

HyperparameterSpace is designed based on Hyperparameter and Condition. HyperparameterSpace can be understood as a container that contains multiple hyperparameters and associations. A DAG framework is implemented in the  HyperparameterSpace. When a DAG is created in HyperparameterSpace, a directed acyclic graph (DAG) is generated. A hyperparameter is added with nodes is related to the DAG. A condition is a directed edge with added to the DAG. When a directed edge is added, validity of the current DAG is detected by using a DAG principle. If it is found that a DAG attribute is not met, an error is reported, indicating a directed acyclic characteristic of an association relationship between hyperparameters.

**The most important function of HyperparameterSpace is sampling.** 
Currently, even random sampling is supported. It is mainly used to provide hyperparameter space for random search or Bayesian optimization. The detailed action is as following: First, obtain the range of each hyperparameter (hyperparameters of the category type are mapped to a continuous space of [0,1]). A matrix with shape of n*d is constructed for each hyperparameter based on the number n of samples(the default value is 1000), d is the number of hyperparameters. Rhe sampled matrix is fed into the subsequent search algorithm for searching, the sampled space may be resampled and updated at any time, thereby ensuring randomness and region coverage of the sample.

### 3.2 Hyperparameter

Overall architecture:

![sha.png](../../images/hyperparameter_space_2.png)

Hyperparameter stores the name, type, and range of each hyperparameter and maps the hyperparameter range to a uniform value range that can be calculated. The hyperparameter types mainly used here are EXP and CAT. The EXP parameters are mapped to [0,1] after the log operation. The CAT parameters are mapped to [0,1] with a catogrized discretation. 

### 3.3 Condition

![sha.png](../../images/hyperparameter_space_3.png)

Condition is used to manage relationships between hyperparameters. Each hyperparameter relationship requires a child and a parent, whether a child is selected depends on whether the current value of the parent meets certain conditions.

Currently, the following conditions are provided: EQUAL, NOT_EQUAL, and IN.  A condition_range is used to transfer the value or range of the condition. The details are as follows:

- EQUAL: condition_range can contain only one parent value, the value of parent must be equal to the value of condition_range.
- NOT_EQUAL class: condition_range can contain one or more parent values,  the parent value must be different from all values provided in condition_range.
- IN: If parent is a range/category type, cond_range must contain two values, indicating the minimum and maximum values. If child is selected, the current value of parent must be within the cond_range.

## 4. User Guide

### 4.1 Example

Asha_hpo is an example to show how to use the HPO module. The example contains the configure file: asha.yaml .

#### Run Pipeline

The main function is to add an HPO pipestep in the following statement:

```bash
vega ./hpo/asha/asha.yml
```

The function is to start the VEGA pipeline for HPO and load the asha.yml configuration file.

#### Configuration file asha.yaml

（1）The example configuration contains the general setting for task and worker to start the pipeline.

```yaml
pipeline: [hpo]
```

This current pipeline contains only one hpo pipestep named hpo.

(2) In the pipestep of hpo,  an HPO configuration part is required to set the HPO algorithm configuration, including total_epochs, config_count, and search space hyperparameter_space.

```yaml
hpo:
    search_algorithm:
        type: AshaHpo
        policy:
            total_epochs: 20

    search_space:
        type: SearchSpace
        hyperparameters:
            -   key: dataset.batch_size
                type: CATEGORY
                range: [8, 16, 32, 64, 128, 256]
            -   key: trainer.optimizer.params.lr
                type: FLOAT_EXP
                range: [0.00001, 0.1]
            -   key: trainer.optimizer.type
                type: CATEGORY
                range: ['Adam', 'SGD']
            -   key: trainer.optimizer.params.momentum
                type: FLOAT
                range: [0.0, 0.99]
        condition:
            -   key: condition_for_sgd_momentum
                child: trainer.optimizer.params.momentum
                parent: trainer.optimizer.type
                type: EQUAL
                range: ["SGD"]
```

config_count indicates the total number of hyperparameter combinations for sampling. In the ASHA algorithm, total_epochs indicates the maximum number of epochs for training one model, hyperparameter_space indicates the current hyperparameter search space, the condition part contains the sub-hyperparameters : "sgd_momentum" is selected only when "parent": "optimizer" is set to "SGD".

(3) The following information needs to be configured for the trainer:

```yaml
hpo:
    model:
        model_desc:
            modules: ["backbone"]
            backbone:
                type: ResNet
                depth: 18
                num_class: 10
    trainer:
        type: Trainer
        epochs: 1
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
```

In addition to the basic configuration of a trainer, the model_desc of a current neural network is also provided, indicating that the neural network is described by the search_space setting in the vega pipeline. Specifically, an image classification network composed of a customized ResNetVariant and LinearClassificationHead is included.

(4) Configure the evaluator.

```yaml
hpo:
    evaluator:
        type: Evaluator
        host_evaluator:
            type: HostEvaluator
            metric:
                type: accuracy
```

Host_evaluator is used to evaluate model performance based on GPU platform and return evaluation results with performance sorting.

#### Running Example

After configuring the customized configuration file, run main.py for the final output.

### 4.2 Module Output

By default, the HPO module generates the score_board.csv, hps.csv, and best_config.json files in the output directory. The output is stored in the corresponding id directory in the worker subdirectory.

#### Scoreboard score_board.csv

The following table lists the values of rung_id for different algorithms, config_id for hyperparameters, running status of the training task corresponding to the ID, and performance score of single-objective optimization.

| rung_id | config_id | status              | score    |
| ------- | --------- | ------------------- | -------- |
| 0       | 0         | StatusType.FINISHED | 1.6      |
| 0       | 1         | StatusType.FINISHED | 12.78261 |
| 0       | 2         | StatusType.FINISHED | 1.2208   |
| 0       | 3         | StatusType.FINISHED | 3.198976 |
| 0       | 4         | StatusType.FINISHED | 12.78772 |

#### ID and hyperparameter combination mapping table hps.csv

Mapping between config_id and hyperparameter combinations. The scoreboard table is as following:

| id   | hps                                                          | performance |
| ---- | ------------------------------------------------------------ | ----------- |
| 0    | {'config_id': 0, 'rung_id': 0, 'configs': {'dataset.batch_size': 64,  'trainer.optim.lr': 0.00014621326777998478, 'trainer.optim.type': 'SGD'},  'epoch': 1} | [1.6]       |
| 1    | {'config_id': 1, 'rung_id': 0, 'configs': {'dataset.batch_size': 256,  'trainer.optim.lr': 2.3729688374364102e-05, 'trainer.optim.type': 'Adam'},  'epoch': 1} | [12.78261]  |
| 2    | {'config_id': 2, 'rung_id': 0, 'configs': {'dataset.batch_size': 16,  'trainer.optim.lr': 0.0006774382480238358, 'trainer.optim.type': 'Adam'},  'epoch': 1} | [1.2208]    |
| 3    | {'config_id': 3, 'rung_id': 0, 'configs': {'dataset.batch_size': 64,  'trainer.optim.lr': 0.009376375563255613, 'trainer.optim.type': 'Adam'},  'epoch': 1} | [3.198976]  |
| 4    | {'config_id': 4, 'rung_id': 0, 'configs': {'dataset.batch_size': 256,  'trainer.optim.lr': 0.016475469254323555, 'trainer.optim.type': 'SGD'},  'epoch': 1} | [12.78772]  |

#### best_config.json

The best hyperparameter selection: parameter combination with the highest score and the current config_id and score are selected as following:

```json
{"config_id": 4,
 "score": 12.78772,
 "configs": {'dataset.batch_size': 256,
             'trainer.optim.lr': 0.016475469254323555,
             'trainer.optim.type': 'SGD'}
}
```

## 5. Appendix

### 5.1 Introduction to the ASHA Algorithm

<https://arxiv.org/abs/1810.05934>

<https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/>

Dynamic resource allocation hyper parameter optimization is used in the SHA algorithm, which is the continuous halving algorithm. The basic idea is as follows: Mmultiple groups of hyperparameters are trained in parallel, and a small number of training iteration are taken in each round. All hyperparameters are evaluated and sorted, all training with hyperparameters arranged in the lower half part is stopped early. A next round of evaluation is performed on the remaining hyperparameters. The evaluation is halved again until the optimization objective is reached.
![sha.png](../../images/sha.png)

The preceding describes the specific operations and ideas of SHA. The problem is that SHA is a serialized or synchronous parallel operation. The next round can be performed only after all hyperparameter training and evaluation in the same round are complete. The asynchronous and parallel algorithm ASHA is proposed for the training accerlation. It performs the next round of evaluation in the current round and continuously synchronizes the growth process, which can be asynchronous and concurrent, greatly improving optimization efficiency.
![asha.png](../../images/asha.png)

### 5.2 HyperBand Algorithm

[Hyperband: Bandit-Based Configuration Evaluation for Hyperparameter Optimization](https://openreview.net/pdf?id=ry18Ww5ee)

![sha.png](../../images/hyperband.png)

- $r$: Budget that can be allocated for a hyperparameter combination;
- $R$: Maximum budget that can be allocated for a hyperparameter combination;
- $s_{max}$: Budget control.
- $B$: Total budget $B=(s_{max}+1)R$；
- $\eta$: Proportion of elimination parameters after each iteration;
- get_hyperparameter_configuration(n): N groups of hyperparameters through sampling;
- run_then_return_val_loss($t$,$r_i$): Valid loss;

**Hyperband example:**

An example on the MNIST dataset is given, and the number of iterations is defined as budget, one budget for each epoch. The hyperparameter search space includes learning rate, batch size, and kernel numbers.

R=81,η=3R=81,η=3, so smax=4,B=5R=5×81smax=4,B=5R=5×81.

The following figure shows the number of hyperparameter groups to be trained and the allocation of hyperparameter resources in each group.

![sha.png](../../images/hyperband_example.png)

There are two level of loops. The inner loop perform the successive halving, number of hyperparameter combinations used for evaluation decreases at each iteration. Meanwhile, the budget that can be allocated to a single hyperparameter combination increases gradually. Therefore, a proper hyperparameter can be found more quickly in this process.

### 5.3 BOHB Algorithm

BOHB is an efficient and stable parameter modulation algorithm proposed by <https://arxiv.org/abs/1807.01774> . Bayesian Optimization (BO) and Hyperband (HB) algorithms are short for BO and HB.

The BOHB depends on the Hyperband (HB) to determine the number of groups of parameters and the amount of resources allocated to each group of parameters. The improvement is that the method of randomly selecting parameters at the beginning of each iteration is replaced by the means of selecting parameters based on the previous data (Bayesian optimization) for parameter selection. Once the number of parameters generated by Bayesian optimization reaches the number of configurations required for iteration, the standard consecutive halving process is started using these configurations.
The performance of these parameters under different resource configurations (budget), g(x, b), is used as the reference data for selecting parameters in next iterations.

### 5.4 BOSS Algorithms

Bayesian Optimization via Sub-Sampling (BOSS) is a general hyperparameter optimization algorithm based on the Bayesian optimization. It is used to efficiently hyperparameter search under the restricted computing resources setting. The core idea is the adaptive allocation of training resources for hyperparameter combination. The final output of the BOSS algorithm is an hyperparameter combination. Under this combination, an optimal model can be obtained after full training. The search of BOSS is as follows:

1. Select the hyperparameters to be searched and the related value ranges.
2. Randomly select a batch of hyperparameter combinations.
3. For these hyperparameter combinations, the Sub-Sampling algorithm is used to allocate computing resources (the number of iterations of the training neural network or the number of samples of the Monte Carlo method) to obtain the corresponding performance indicators of each combination.
4. The Bayesian model (TPE model) is updated based on the newly added combinations, and the next batch of combinations are extracted from the updated  TPE model.
5. Repeat steps 3-4 until the max iterations are reached or predefined performance is achieved.

### 5.5 TPE Algorithm

TPE is based on Bayesian ideas. Different from GP, TPE simulates p(x|y) instead of p(y|x) in the modeling loss function. See the reference <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf> for detailed algorithm description.
