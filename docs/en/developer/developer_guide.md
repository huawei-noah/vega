# Development Reference

## 1. Introduction

The key features of Vega are network architecture search and hyperparameter optimization. In the network architecture search process, the search space and search algorithm are the core parts, and the generator is used to control the sampling, update, and end of the search process.

The following figure shows the class diagram of the search space and search algorithm.

![Search Space diagram](./images/search_space_classes.png)

The following figure shows the search space and search algorithm process.

![Search Space process](./images/search_space_flow.png)

Search space process
The following describes the following parts:

- search space
- search algorithm
- Complete NAS search process
- Configuration
- Pipeline
- Trainer and fully train

## 2. Search space

### 2.1 Networks and NetworksFactory

The search space of the Vega NAS includes a series of predefined network types, and a combination of these networks or a construction parameter of each network may be used as the search space of the NAS.

Various network types are defined under vega/search_space/networks, and are classified into backbone, head, RPN, blocks, and super_network based on functions. The backbone refers to the backbone part of the classification or detection network, and the head refers to the output header of the classification or detection network, blocks are basic operations or common modules consisting of basic operations. super_network is a super network commonly used in Parameter Sharing. Custom networks that do not belong to the preceding definition exist.

- search_space
  - netowrks
    - backbones
    - blocks
    - heads
    - rpn
    - super_network
    - customs

The Vega predefined and new network initialization parameters both accept only one parameter in the dict format, which is a complete description of the network. The initialization function parses the description and generates a network model. For Vega, the search space of a network is the value range of the description.

After the network class is defined, register the networks with NetworkFactory according to their types. Take the network registration of ResNet as an example. The code is as follows:

```python
@NetworkFactory.register(NetTypes.BACKBONE)
class ResNet(Network):
    pass
```

As shown in the preceding code, add a registration statement of NetworkFactory before the definition of the ResNet class. Because ResNet belongs to the backbone type, register it as the NetTypes.BACKBONE type.

### 2.2 Configuration Files

After the network class is defined and registered, you need to write the description of the network class construction into the configuration file and provide the optional ranges of some parameters. If the search range of some parameters is not specified in the configuration file, you can leave the description blank. However, the complete parameters must be included in the network description. The following uses the PruneResNet configuration file as an example:

```yaml
search_space:
    type: SearchSpace
    modules: ['backbone']
    backbone:
        type: 'PruneResNet'
        base_chn: [16,16,16,32,32,32,64,64,64]
        base_chn_node: [16,16,32,64]
        num_classes: 10
```

The configuration file does not contain all parameters. The search algorithm needs to calculate and parse these parameters to obtain other parameters before generating the PruneResNet description as the PruneResNet construction parameters. For details, see the example in vega/algorithms/compression/prune_ea.

### 2.3 SearchSpace and NetworkDesc

When the SearchSpace class is initialized, the search_space attribute is loaded as follows:

```python
class SearchSpace(object):
    def __init__(self):
        self.search_space = self.cfg
```

Another important concept of the search space is the network description. The network description is the result sampled by the search algorithm from the search space and is a possible subset in the search space. The network description class has only one attribute, that is, the network description of the dict type (one or multiple networks). The network description class has only one general interface to_model(), which is responsible for analyzing the network description and automatically resolving the network description into the specific network object in the Networks through NetworFactory.

```python
class NetworkDesc(object):
    def __init__(self, desc):
        self.desc = Config(desc)

    def to_model(self):
        pass
```

In general, Vega provides a series of network models (added by developers) and registers them with NetworkFactory. Developers need to write the search space of the network model construction parameters in the configuration file, and use the algorithm to sample and generate the network description NetworkDesc. NetworkDesc automatically parses the corresponding network model.

## 3. Search algorithm

The SDK of Vega provides some default search algorithms, such as random search and basic evolution algorithms. Developers can extend the search algorithms as required. These algorithms need to be registered with the unified ClassFactory, and the label is ClassType.SEARCH_ALGORITHM.

There are two parameters for initializing the search algorithm. One is the search_algorithm part in the configuration file, and the other is the object of the SearchSpace class.

The search algorithm provides the following functions:

- Search for a network description net_desc from the search space.
- Update the search algorithm based on the training result.
- Check whether the search process is complete.

The most important of these is the first feature, which is responsible for searching a subset of the SearchSpace objects for a network description.

```python
class SearchAlgorithm(TaskUtils):

    def __init__(self, search_space=None, **kwargs):
        super(SearchAlgorithm, self).__init__(self.cfg)

    def search(self):
        raise NotImplementedError

    def update(self, local_worker_path):
        raise NotImplementedError

    @property
    def is_completed(self):
        return False
```

Some algorithms (such as EA) may also involve the coding of the search space. Therefore, a codec needs to be implemented in the search algorithm. The codec mainly implements two functions: coding the network description, and decoding the code into the network description.

```python
class Codec(object):
    def encode(self, desc):
        raise NotImplementedError

    def decode(self, code):
        raise NotImplementedError
```

The types and parameters of search algorithms need to be written in the configuration file. The following uses PruneEA as an example:

```yaml
search_algorithm:
    type: PruneEA
    length: 464
    num_generation: 31
    num_individual: 4
    metric_x: flops
    metric_y: acc
    random_models: 32
    codec: PruneCodec
```

In the configuration file, you need to define the type of the search algorithm and the parameters of the search algorithm.

## 4. NAS search process

The search process of NAS mainly includes two parts: generator and trainer. The generator samples a network model in the search space by using the search algorithm, initializes the network model as a trainer, and distributes the trainer to nodes for running.

The NAS search process is completed in NasPipeStep. The main function of NasPipeStep is completed in the do() function. The code is as follows:

```python
def do(self):
        """Do the main task in this pipe step."""
        logger.info("NasPipeStep started...")
        while not self.generator.is_completed:
            id, model = self.generator.sample()
            cls_trainer = ClassFactory.get_cls('trainer')
            trainer = cls_trainer(model, id)
            self.master.run(trainer)
            finished_trainer_info = self.master.pop_finished_worker()
            self.update_generator(self.generator, finished_trainer_info)
        self.master.join()
        finished_trainer_info = self.master.pop_all_finished_train_worker()
        self.update_generator(self.generator, finished_trainer_info)
```

In each cycle, the generator first determines whether the search stops. If the search stops, the search ends, the generator is updated, and a value is returned.

If the Trainer object is not stopped, the Generator generates a network model and ID through the sample() function. The ClassFactory locates the specific Trainer class according to the configuration file, and then initializes the Trainer object through the network model and Trainer configuration parameters. The master distributes the trainer to an idle node, obtains the trainer result of the completed node, and then updates the generator, for example, in a loop.

### 4.1 Generator

The generator defines the Search Space and Search Algorithm objects. In each subsequent loop, the Search Algorithm samples a model from the Search Space and initializes the model as the Trainer of the NAS.

This is a standard procedure and does not require additional or reimplementation unless there are special processing steps. The generator implementation code is as follows:

```python
class Generator(object):
    _subclasses = {}

    def __init__(self):
        self.search_space = SearchSpace()
        self.search_alg = SearchAlgorithm(self.search_space)

    @property
    def is_completed(self):
        return self.search_alg.is_completed

    def sample(self):
        id, net_desc = self.search_alg.search()
        model = net_desc.to_model()
        return id, model

    def update(self, worker_path):
        self.search_alg.update(worker_path)
        return
```

During initialization, the search space object is generated in the search_space part of the configuration file, and the search space is used as the parameter to initialize the search algorithm object.

The sample interface in the code is used for each sampling in the NAS. The sample interface first invokes the search algorithm to search for a network description, and then generates a network model based on the network description.

In addition, the generator can determine whether the iterative search stops and update the search algorithm.

### 4.2 Trainer

In NasPipeStep, after the generator generates a network model, a trainer is initialized. The trainer is a complete full trainer process, and its main interfaces are train_process, some standard interfaces such as optimizers, learning rate policies, and loss functions. Vega provides the standard Trainer interface and training process. Developers only need to modify the configuration file to control the training parameters and training process. You can also customize some functions that are not provided.

The trainer configuration is as follows:

```yaml
trainer:
    type: Trainer
    optim:
        type: SGD
        lr: 0.1
        momentum: 0.9
        weight_decay: !!float 1e-4
    lr_scheduler:
        type: StepLR
        step_size: 20
        gamma: 0.5
    loss:
        type: CrossEntropyLoss
    metric:
        type: accuracy
    report_freq: 50
    epochs: 50
```

The trainer configuration parameters need to write the names and parameters of the optimizer, learning rate policy, and loss function to the corresponding positions. The standard trainer provides the initialization interface for parsing these objects.

The standard trainer training process is implemented in the train_process interface. The implementation is as follows:

```python
 def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        self._init_estimator()
        self._init_dataloader()
        logging_hook = []
        if self.horovod:
            logging_hook += [hvd.BroadcastGlobalVariablesHook(0)]
        train_steps = self.train_data.data_len
        valid_steps = self.valid_data.data_len
        if self.horovod:
            train_steps = train_steps // hvd.size()
            valid_steps = valid_steps // hvd.size()
        start_step = est._load_global_step_from_checkpoint_dir(self.cfg.model_dir)
        for i in range(self.cfg.epochs):
            logging.info('train epoch [{0}/{1}]'.format(i, self.cfg.epochs))
            current_max_step = start_step + train_steps
            start_step = current_max_step
            self.estimator.train(input_fn=self.train_data.input_fn,
                                 max_steps=current_max_step,
                                 hooks=logging_hook)
            eval_results = self.estimator.evaluate(input_fn=self.valid_data.input_fn, steps=valid_steps)
            logging.info(eval_results)
        self.save_backup(eval_results)
```

To facilitate developers, we encapsulate some capabilities to be used in the trainer and provide extension interfaces for the.

#### Optimizer

By default, the torch.optim file in the Pytroch library is used. The file is directly used in configuration mode. type indicates the method to be used. Other key values are the input parameters and their values in the method.

```yaml
optim:
        type: SGD
        lr: 0.1
        momentum: 0.9
        weight_decay: !!float 1e-4
```

#### Loss

By default, all loss functions in the torch.nn file can be directly used in configuration mode. type indicates the method to be used. Other key values are the input parameters and values of the input parameters in the method.

```yaml
loss:
    type: CrossEntropyLoss
```

You can also customize the Loss function and specify it in the configuration.

- Use @ClassFactory.register(ClassType.LOSS) for registration.

```python

@ClassFactory.register(ClassType.LOSS)
class CustomCrossEntropyLoss(Network):
    """Cross Entropy Loss."""

    def __init__(self, desc):
        super(CustomCrossEntropyLoss, self).__init__()
            pass

    def forward(self, **kwargs):
        pass

```

- Reference CustomCrossEntropyLoss in the configuration file.

```yaml
loss:
        type: CustomCrossEntropyLoss
        desc: ~
```

#### LrScheduler

By default, all lr_scheduler functions in torch.optim.lr_scheduler can be directly used in configuration mode. type indicates the method to be used. Other key values are the values of the input parameters in the method.

```yaml
lr_scheduler:
        type: StepLR
        step_size: 20
        gamma: 0.5
```

Customize an LrScheduler.

- Register with @ClassFactory.register(ClassType.LOSS) and reference it in the configuration file.
- The step interface needs to be implemented. The input parameter is epoch.

```python
@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler(_LRScheduler):
    def step(self, epoch=None):
         pass
```

#### Metrics

Common metrics are preset in VEGA and can be configured in the configuration file. Multiple metrics can be processed for printing and analysis. When there are multiple metrics, the first metric function is automatically used to calculate the loss.

```yaml
metric:
   type: accuracy
```

Customize a metric.

- Use **@ClassFactory.register(ClassType.METRIC)** for registration.
- Inherited from **vega.core.metrics.metrics_base.MetricsBase**
- Specify __metric_name__ for recording and printing metrics.
- Implement the __call__ and **summay** and **reset** methods. The call method is invoked at each step, and the summay method is invoked after each epoch.

```python
@ClassFactory.register(ClassType.METRIC, alias='accuracy')
class Accuracy(MetricBase):
    """Calculate classification accuracy between output and target."""

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1,)):
        """Init Accuracy metric."""
        self.topk = topk
        self.sum = [0.] * len(topk)
        self.data_num = 0
        self.pfm = [0.] * len(topk)

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        if isinstance(output, tuple):
            output = output[0]
        res = accuracy(output, target, self.topk)
        n = output.size(0)
        self.data_num += n
        self.sum = [self.sum[index] + item.item() * n for index, item in enumerate(res)]
        self.pfm = [item / self.data_num for item in self.sum]
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = [0.] * len(self.topk)
        self.data_num = 0
        self.pfm = [0.] * len(self.topk)

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
```

#### Customized Trainer

If the common trainers provided by Huawei cannot meet the current requirements, you can use the following methods to customize the trainers:

- Use the @ClassFactory.register(ClassType.TRAINER) for registration.
- Inherit the vega.core.trainer.trainer.Trainer base class.
- The train_process method is overwritten.
from vega.core.trainer.trainer import Trainer
from vega.core.common.class_factory import ClassFactory, ClassType

```python
from vega.core.trainer.trainer import Trainer
from vega.core.common.class_factory import ClassFactory, ClassType

@ClassFactory.register(ClassType.TRAINER)
class BackboneNasTrainer(Trainer):

    def __init__(self, model, id):
        """Init BackboneNasTrainer."""
        super(BackboneNasTrainer, self).__init__(model, id)
        self.best_prec = 0

    def train_process(self):
        pass
```

> Note: We can override finer - grained methods like tain and valid so that we can use some of the capabilities provided by the trainer base class.

## 5. Configuration

The Vega Configuration uses the registration mechanism. It dynamically maps the configuration file to the corresponding instance based on the class type. In this way, developers and users can directly use the cfg attribute without being aware of the loading and parsing process of the configuration file.

The following describes how to use the configuration mechanism:

- **Step1: Use rega.run() to load the user-defined configuration file and run the VEGA program.**

    ```python
    vega.run('config.yml')
    ```

- **Step2: Use the following definition in the config.yml file**

    ```yaml
    # Common configuration, including task and worker configuration information. 
    general:
        task:
            key: value
        worker:
            key: value
    # Pipestep execution sequence
    pipeline: [nas1, fullytrain1]
    # pipestep name
    nas1:
        pipe_step:
            type: NasPipeStep
        search_algorithm:
            type: BackboneNas
            key: value
        search_space:
            type: SearchSpace
            key: value
        mode:
            model_desc: value
        trainer:
            type: Trainer
        dataset:
            type: Cifar10
    ```

- **Step3: Use ClassFactory to register the class that needs to be configured.**

    The ClassFactory provides multiple ClassType options for developers, which correspond to the level-2 nodes in the config file.

    ```python
    
    class ClassType(object):
        """Const class saved defined class type."""
    
        TRAINER = 'trainer'
        METRIC = 'trainer.metric'
        OPTIM = 'trainer.optim'
        LR_SCHEDULER = 'trainer.lr_scheduler'
        LOSS = 'trainer.loss'
        EVALUATOR = 'evaluator'
        GPU_EVALUATOR = 'evaluator.gpu_evaluator'
        HAVA_D_EVALUATOR = 'evaluator.hava_d_evaluator'
        DAVINCI_MOBILE_EVALUATOR = 'evaluator.davinci_mobile_evaluator'
        SEARCH_ALGORITHM = 'search_algorithm'
        SEARCH_SPACE = 'search_space'
        PIPE_STEP = 'pipe_step'
        GENERAL = 'general'
        HPO = 'hpo'
        DATASET = 'dataset'
        TRANSFORM = 'dataset.transforms'
        CALLBACK = 'trainer.callback'
    ```

    The algorithm developer selects the corresponding ClassType as required and uses @ClassFactory.register(class type) to register the class to the corresponding class. In the following example, the BackboneNas is registered with ClassType.SEARCH_ALGORITHM. The Configuration module determines to initialize the BackboneNas based on the value of type under search_algorithm in config.yml and binds the configuration information to the cfg attribute of the BackboneNas.

    Developers can directly use the self.cfg attribute as follows:

    ```python
    @ClassFactory.register(ClassType.SEARCH_ALGORITHM)
    class BackboneNas(SearchAlgorithm):
        def __init__(self, search_space=None):
            """Init BackboneNas."""
            super(BackboneNas, self).__init__(search_space)
            # ea or random
            self.search_space = search_space
            self.codec = Codec(self.cfg.codec, search_space)
            self.num_mutate = self.policy.num_mutate
            self.random_ratio = self.policy.random_ratio
            self.max_sample = self.range.max_sample
            self.min_sample = self.range.min_sample
    ```

- **Step4: The developer needs to provide the default configuration. The user configuration overwrites the default configuration.**

    It is recommended that each developer provide a default configuration file for the system when compiling an algorithm. This helps users configure their own configuration files.

    In the vega.config directory, you can group directories. The default configuration file must be stored in the corresponding directory.

    ```text
    vega/config
    ├── datasets
    │   └── cifar10.yml
    ├── general
    │   └── general.yml
    ├── search_algorithm
    │   └── backbone.yml
    ├── search_space
    │   └── search_space.yml
    └── trainer
        └── trainer.yml
    ```

    The default configuration uses the key:value format. The root key value corresponds to the full name of the class defined by the developer.

    ```yaml
    BackboneNas:
        codec: BackboneNasCodec
        policy:
            num_mutate: 10
            random_ratio: 0.2
        range:
            max_sample: 100
            min_sample: 10

    ```

## 6. pipeline

The pipeline of Vega implements the concatenation of multiple pipelines by loading the config configuration. When the user executes the vega.run('config.yml') method, the _init_env(cfg_path) method is executed to load the configuration, and then the pipeline().run() method is invoked. Run the pipestep do() function according to the definition in the configuration file.

### 6.1 Configuration

In the config.yml file, pipeline is used to define the pipestep execution sequence. In the following example, pipeline: [nas, fully train] indicates that the pipestep of the NAS node is executed first, and then the pipestep of the fully train node is executed.

```yaml
# Define pipelines and execute the pipestep
pipeline: [nas, fullytrain]
# pipestep name
nas:
# PipeStep type
    pipe_step:
        type: NasPipeStep

fullytrain:
    pipe_step:
        type: FullyTrainPipeStep
```

### 6.2 Extended Pipestep

The currently preset `pipestep` are:

Currently, the following pipesteps have been preset:

- NasPipeStep
- HpoPipeStep
- FullyTrainPipeStep

To extend pipestep, inherit the base class PipeStep and implement the do() function. For details, see the implementation code of the preceding class.

```python
class PipeStep(object):

    def __init__(self):
        self.task = TaskUtils(UserConfig().data.general)

    def __new__(cls):
        """Create pipe step instance by ClassFactory"""
        t_cls = ClassFactory.get_cls(ClassType.PIPE_STEP)
        return super().__new__(t_cls)

    def do(self):
        """Do the main task in this pipe step."""
        raise NotImplementedError
```

## 7. Fully Train

On `Fully Train`, we support single-card training and multi-machine multi-card distributed training based on `Horovod`. `Fully Train` corresponds to `FullyTrainPipeStep` part of `pipeline`.

### 7.1 Configuration

If you need to conduct distributed training of `Horovod`, you need to add a configuration item `horovod` to the configuration file in the `trainer` part of `FullyTrainPipeStep` and set it to `True`. If there is no such item, the default is False, ie Does not use distributed training.

```yaml
fullytrain:
    pipe_step:
        type: FullyTrainPipeStep
    trainer:
        type: trainer
        horovod: True
```

We started the `Horovod` distributed training through the `shell`, and the communication configuration between different nodes has been completed in the mirror. The developer can not care about how the `vega` is started internally.

### 7.2 Trainer supports Horovod distribution

When using distributed training, in contrast to single-card training, the trainer's network model, optimizer, and data loading need to be packaged into distributed objects using `Horovod`.

```python
def _init_optimizer(self):
    ...
    if self.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=self.model.named_parameters(),
                                             compression=hvd.Compression.none)
    return optimizer

def _init_horovod_setting(self):
    """Init horovod setting."""
    hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

def _init_dataloader(self):
    """Init dataloader."""
    train_dataset = Dataset(mode='train')
    valid_dataset = Dataset(mode='test')
    if self.horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_dataset.sampler = train_sampler
        valid_dataset.sampler = valid_sampler
    self.train_loader = train_dataset.dataloader
    self.valid_loader = valid_dataset.dataloader
```

In the process of training, the codes of single card and distributed training are almost the same, but when the verification index is finally calculated, the index values on different cards need to be combined to calculate the total average.

```python
def _metric_average(self, val, name):
    """Do metric average.

    :param val: input value
    :param name: metric name
    :return:
    """
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()
```
