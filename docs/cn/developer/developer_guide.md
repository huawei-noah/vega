# 开发参考

## 1. Vega简介

Vega的重点特性是网络架构搜索和超参优化，在网络架构搜索流程中，搜索空间`Search Space`、搜索算法`Search Algorithm`是核心部分，并通过`Generator`来控制搜索的采样、更新和结束等流程步骤。

搜索空间和搜索算法的类图如下所示：

![Search Space类图](../../images/search_space_classes.png)

搜索空间和搜索算法的流程图如下所示：

![Search Space流程图](../../images/search_space_flow.png)

以下就分别介绍下面几个部分：

* 搜索空间
* 搜索算法
* 完整的NAS搜索流程
* `Configuration`
* `Pipeline`
* `Trainer 和 fully train`

## 2. 搜索空间

### 2.1 Networks和NetworksFactory

Vega NAS的搜索空间包含了一系列预定义的网络类型，这些网络的组合或者每个网络的构造参数均可作为NAS的搜索空间。

在`vega/search_space/networks`下面定义了丰富的网络类型，并按照功能分成`backbone`、`head`、`RPN`、`blocks`、`super_network`等等，其中`backbone`指的是分类或者检测等网络中的骨干部分，`head`指的是分类或者检测等网络中输出头部分，`blocks`是一些基本的操作或者是由基本操作构成的一些常用的模块，`super_network`是`Parameter Sharing`里常用的超级网络等。还有不属于以上定义的自定义网络`customs`。

* search_space
  * netowrks
    * backbones
    * blocks
    * heads
    * rpn
    * super_network
    * customs

Vega的预定义和新增网络初始化参数均只接受一个参数，该参数的格式是`dict`，它是对这个网络的一个完整描述，在初始化函数里，将这些描述解析并生成网络模型。对于Vega来说，一个网络的`Search Space`就是这些描述的取值范围。

定义网络类后，将这些网络按照各自的类型注册到`NetworkFactory`，以`ResNet`的网络注册为例，代码如下：

```python
@NetworkFactory.register(NetTypes.BACKBONE)
class ResNet(Network):
    pass
```

如上代码所示，在`ResNet`类定义前面增加一个`NetworkFactory`的注册语句，因为`ResNet`属于`backbone`的类型，所以将它注册为`NetTypes.BACKBONE`的类型。

### 2.2 配置文件

在定义好网络类和注册之后，开发者需要将网络类构造描述详细地写在配置文件里，并给出某些参数的可选范围，如果某些参数不好通过配置文件给出搜索范围，可以先不写，但是在搜索出来的网络描述里一定要包括完整的参数。以`PruneResNet`的配置文件为例：

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

这里的配置文件并没有包括所有的参数，搜索算法还需要对这些参数进行计算和解析，得到另外的几个参数后，才能一起生成`PruneResNet`的描述，作为`PruneResNet`的构造参数，具体请参考`vega/algorithms/compression/prune_ea`的示例。

### 2.3 SearchSpace类和NetworkDesc类

`SearchSpace`类初始化时，加载搜索空间配置文件为`search_space`属性，如下：

```python
@ClassFactory.register(ClassType.NETWORK)
class SearchSpace(object):

    config = SearchSpaceConfig()

    def __new__(cls, *args, **kwargs):
        t_cls = ClassFactory.get_cls(ClassType.NETWORK)
        return super(SearchSpace, cls).__new__(t_cls)

    @property
    def search_space(self):
        return self.config.to_dict()
```

搜索空间还有一个重要的概念是网络描述`NetworkDesc`，网络描述是搜索算法从`SearchSpace`里采样出来的结果，它是`Search Space`中的一种可能性子集。网络描述类里只有一个属性，就是dict类型的网络描述（可以是一个网络或者多个网络）。网络描述类只一个通用的`to_model()`的接口，负责分析网络描述并通过`NetworFactory`自动解析成`Networks`里具体的网络对象。

```python
class NetworkDesc(object):

    def __init__(self, desc):
        self._desc = Config(deepcopy(desc))
        self._model_type = None
        self._model_name = None

    def to_model(self):
        model = FineGrainedNetWork(self._desc).to_model()
        if model is not None:
            return model
        networks = []
        module_types = self._desc.get('modules')
        for module_type in module_types:
            network = self.to_coarse_network(module_type)
            networks.append((module_type, network))
        if len(networks) == 1:
            return networks[0][1]
        else:
            if vega.is_torch_backend():
                import torch.nn as nn
                networks = OrderedDict(networks)
                return nn.Sequential(networks)
            elif vega.is_tf_backend():
                from .tensorflow import Sequential
                return Sequential(networks)
```

总的来说，Vega提供预定义（支持开发者新增）的一系列网络模型`Networks`，并注册到`NetworkFactory`，开发者需要将网络模型构造参数的搜索空间写在配置文件中，通过算法去采样和生成网络描述`NetworkDesc`，`NetworkDesc`自动解析出相应的网络模型。

## 3. 搜索算法

Vega的SDK提供一些默认的搜索算法，比如随机搜索、基本进化算法等，开发者可根据需要自行扩充搜索算法。这些算法都需要注册到统一的`ClassFactory`里，标签为`ClassType.SEARCH_ALGORITHM`。

搜索算法的初始化有两个参数，一个是配置文件中的`search_algorithm`部分， 还有一个是`SearchSpace`类的对象。

搜索算法主要完成的功能有：

* 从搜索空间中搜索出一个网络描述`net_desc`
* 从训练结果中去更新搜索算法
* 判断搜索过程是否完成

其中，最重要的是第一个功能，它负责在`SearchSpace`对象中搜索出一个子集，作为网络描述。

```python
class SearchAlgorithm(TaskOps):

    def __new__(cls, *args, **kwargs):
        t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM)
        return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        super(SearchAlgorithm, self).__init__()
        self.search_space = search_space
        self.codec = Codec(search_space, type=self.config.codec)

    def search(self):
        raise NotImplementedError

    def update(self, record):
        pass

    @property
    def is_completed(self):
        raise NotImplementedError
```

在一些算法中（比如`EA`），可能还会涉及到搜索空间的编码问题，所以还要在搜索算法里实现一个编解码器`Codec`，编解码器主要完成两个功能，一个是将网络描述编码化，还有一个是将编码解码成网络描述。

```python
class Codec(object):
    def encode(self, desc):
        raise NotImplementedError

    def decode(self, code):
        raise NotImplementedError
```

搜索算法的类别和参数需要写在配置文件中，以`PruneEA`为示例：

```yaml
search_algorithm:
    type: PruneEA
    codec: PruneCodec
    policy:
        length: 464
        num_generation: 31
        num_individual: 32
        random_samples: 64
```

配置文件中，需要定义搜索算法的类型，以及该类型搜索算法的参数。

## 4. NAS搜索流程

NAS的搜索流程主要包括`Generator`和`Trainer`两个部分，其中`Generator`负责通过搜索算法在搜索空间中采样出一个网络模型，将网络模型初始化成`Trainer`后，`Trainer`被分发到节点上运行。

NAS的搜索流程是在`SearchPipeStep`中完成的，`SearchPipeStep`的主要功能是在`do()`函数中完成的，实现代码如下：

```python
def do(self):
    while not self.generator.is_completed:
        res = self.generator.sample()
        if res:
            self._dispatch_trainer(res)
        else:
            time.sleep(0.5)
        self._after_train(wait_until_finish=False)
    self.master.join()
    self._after_train(wait_until_finish=True)
    ReportServer().output_pareto_front(General.step_name)
    self.master.close()
```

在每一次循环中，`Generator`首先判断搜索是否停止，如果停止了就结束搜索，更新`Generator`并返回。

如果未停止，`Generator`通过`sample()`函数生成一个网络模型和`id`，`ClassFactory`根据配置文件定位到具体的`Trainer`类，再通过网络模型和`Trainer`配置参数初始化出对应的`Trainer`对象。`Master`负责将`trainer`分发到空闲的节点上运行，并得到已完成节点的`trainer`结果，再去更新`Generator`，如是循环。

### 4.1 Generator

Generator里将定义Search Space和Search Algorithm的对象，后面在每一次循环中，Search Algorithm从Search Space中采样出一个model，并将model初始化成NAS的Trainer。
这是一个标准的过程，如果没有特殊的处理步骤，无需额外添加或者重新实现。Generator的实现代码如下：

```python
class Generator(object):
    _subclasses = {}

    def __init__(self):
        self.search_space = SearchSpace()
        self.search_alg = SearchAlgorithm(self.search_space.search_space)

    @property
    def is_completed(self):
        return self.search_alg.is_completed

    def sample(self):
        id, desc = self.search_alg.search()
        return id, desc

    def update(self, step_name, worker_id):
        record = reportClinet.get_record(step_name, worker_id)
        logging.debug("Get Record=%s", str(record))
        self.search_alg.update(record.serialize())
```

初始化时，首先通过配置文件中search_space部分生成搜索空间的对象，将搜索空间作为参数初始化搜索算法的对象。
代码中的sample接口即是NAS中每一次采样，首先调用搜索算法search出一个网络描述，再通过网络描述生成网络模型。
此外，Generator还具有判断迭代搜索是否停止以及更新搜索算法等功能。

## 5 Trainer

Trainer用于训练模型，在NAS、HPO、fully train等阶段，可将trainer配置这些阶段的pipestep中，完成模型的训练。

一般在配置文件中，Trainer的配置形式如下所示：

```yaml
trainer:
    type: Trainer
    optim:
        type: SGD
        params:
            lr: 0.1
            momentum: 0.9
            weight_decay: !!float 1e-4
    lr_scheduler:
        type: StepLR
        params:
            step_size: 20
            gamma: 0.5
    loss:
        type: CrossEntropyLoss
    metric:
        type: accuracy
    report_freq: 50
    epochs: 50
```

trainer的主要函数是train_process()，该函数定义如下：

```python
    def train_process(self):
        self._init_callbacks(self.callbacks)
        self._train_loop()

    def _init_callbacks(self, callbacks):
        self.callbacks = CallbackList(self.config.callbacks, disables)
        self.callbacks.set_trainer(self)

    def _train_loop(self):
        self.callbacks.before_train()
        for epoch in range(self.epochs):
            epoch_logs = {'train_num_batches': len(self.train_loader)}
            if self.do_validation:
                epoch_logs.update({'valid_num_batches': len(self.valid_loader)})
            self.callbacks.before_epoch(epoch, epoch_logs)
            self._train_epoch()
            if self.do_validation and self._should_run_validation(epoch):
                self._valid_epoch()
            self.callbacks.after_epoch(epoch)
        self.callbacks.after_train()

    def _train_epoch(self):
        if vega.is_torch_backend():
            self.model.train()
            for batch_index, batch in enumerate(self.train_loader):
                batch = self.make_batch(batch)
                batch_logs = {'train_batch': batch}
                self.callbacks.before_train_step(batch_index, batch_logs)
                train_batch_output = self.train_step(batch)
                batch_logs.update(train_batch_output)
                if self.config.is_detection_trainer:
                    batch_logs.update({'is_detection_trainer': True})
                self.callbacks.after_train_step(batch_index, batch_logs)
        elif vega.is_tf_backend():
            self.estimator.train(input_fn=self.train_input_fn,
                                 steps=len(self.train_loader),
                                 hooks=self._init_logging_hook())

    def _valid_epoch(self):
        self.callbacks.before_valid()
        valid_logs = None
        if vega.is_torch_backend():
            self.model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(self.valid_loader):
                    batch = self.make_batch(batch)
                    batch_logs = {'valid_batch': batch}
                    self.callbacks.before_valid_step(batch_index, batch_logs)
                    valid_batch_output = self.valid_step(batch)
                    self.callbacks.after_valid_step(batch_index, valid_batch_output)
        elif vega.is_tf_backend():
            eval_metrics = self.estimator.evaluate(input_fn=self.valid_input_fn,
                                                   steps=len(self.valid_loader))
            self.valid_metrics.update(eval_metrics)
            valid_logs = dict()
            valid_logs['cur_valid_perfs'] = self.valid_metrics.results
        self.callbacks.after_valid(valid_logs)
```

从以上代码可以看出，trainer使用了callback机制，将模型的训练过程中插入了before_train()、before_epoch()、before_train_step()、after_train_step()、after_epoch()、before_valid()、before_valid_step()、after_valid_step()、after_valid()、after_train()这十个插入点，用户根据需要，定制callback，完成特定的模型训练过程。

同时Vega提供了缺省的Callback：

- pytorch：ModelStatistics、MetricsEvaluator、ModelCheckpoint、PerformanceSaver、LearningRateScheduler、ProgressLogger、ReportCallback
- TensorFlow：ModelStatistics、MetricsEvaluator、PerformanceSaver、ProgressLogger、ReportCallback

### 5.1 Optimizer

默认使用pytorch库上的`torch.optim`，采用配置方式直接使用，`type`表示使用的方法，其他键值为方法中的入参和入参的值

```yaml
optim:
    type: SGD
    params:
        lr: 0.1
        momentum: 0.9
        weight_decay: !!float 1e-4
```

### 5.2 Loss

默认可以直接使用`torch.nn`下的所有loss函数，采用配置方式使用，`type`表示使用的方法，其他键值为方法中的入参和入参的值

```yaml
loss:
    type: CrossEntropyLoss
```

也可以自定义Loss函数并在配置中指定:

- 使用`@ClassFactory.register(ClassType.LOSS)`进行注册

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

- 在配置文件中引用CustomCrossEntropyLoss

```yaml
loss:
    type: CustomCrossEntropyLoss
    desc: ~
```

### 5.3 LrScheduler

​	默认可以直接使用`torch.optim.lr_scheduler`下的所有lr_scheduler函数，采用配置方式使用，`type`表示使用的方法，其他键值为方法中的入参和入参的值

```yaml
lr_scheduler:
    type: StepLR
    step_size: 20
    gamma: 0.5
```

自定义一个LrScheduler

- 使用`@ClassFactory.register(ClassType.LOSS)`进行注册，并在配置文件中引用
- 需要实现step接口，入参为epoch

```python
@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler(_LRScheduler):
    def step(self, epoch=None):
         pass
```

### 5.4 Metrics

常用的metrics已预置在vega中，可直接在配置文件中配置使用，同时支持处理多个metrics进行打印分析。当有多个metrics的时候，会自动以第一个metric函数计算loss。

```yaml
metric:
   type: accuracy
```

自定义一个metric

- 使用`@ClassFactory.register(ClassType.METRIC)`进行注册
- 继承`vega.metrics.metrics_base.MetricBase`
- 指定`__metric_name__`，供记录打印metrics使用
- 实现`__call__`、`summay`、`reset`方法，call是在每轮step的时候调用，summay是每轮epoch结束后调用

```python
@ClassFactory.register(ClassType.METRIC, alias='accuracy')
class Accuracy(MetricBase):

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1,)):
        self.topk = topk
        self.sum = [0.] * len(topk)
        self.data_num = 0
        self.pfm = [0.] * len(topk)

    def __call__(self, output, target, *args, **kwargs):
        if isinstance(output, tuple):
            output = output[0]
        res = accuracy(output, target, self.topk)
        n = output.size(0)
        self.data_num += n
        self.sum = [self.sum[index] + item.item() * n for index, item in enumerate(res)]
        self.pfm = [item / self.data_num for item in self.sum]
        return res

    def reset(self):
        self.sum = [0.] * len(self.topk)
        self.data_num = 0
        self.pfm = [0.] * len(self.topk)

    def summary(self):
        if len(self.pfm) == 1:
            return self.pfm[0]
        return {'top{}_{}'.format(self.topk[idx], self.name): value for idx, value in enumerate(self.pfm)}
```

另外，我们支持多个metrics的统一管理，使用Metrics类管理trainer各种不同类型的metrics，统一各个metrics的初始化、调用的接口和获取结果的方式。

- 根绝配置文件中metric部分的内容初始化`Metrics`，在每次valid的时候初始化一次
- 根据网络的输出数据和数据集标签数据，调用`__call__`，计算每次数据的metric和历史平均值
- `results`返回metric的历史综合结果

```python
class Metrics(object):

    config = MetricsConfig()

    def __init__(self, metric_cfg=None):
        """Init Metrics."""
        self.mdict = {}
        metric_config = obj2config(self.config) if not metric_cfg else deepcopy(metric_cfg)
        if not isinstance(metric_config, list):
            metric_config = [metric_config]
        for metric_item in metric_config:
            ClassFactory.get_cls(ClassType.METRIC, self.config.type)
            metric_name = metric_item.pop('type')
            metric_class = ClassFactory.get_cls(ClassType.METRIC, metric_name)
            if isfunction(metric_class):
                metric_class = partial(metric_class, **metric_item.get("params", {}))
            else:
                metric_class = metric_class(**metric_item.get("params", {}))
            self.mdict[metric_name] = metric_class
        self.mdict = Config(self.mdict)

    def __call__(self, output=None, target=None, *args, **kwargs):
        pfms = []
        for key in self.mdict:
            metric = self.mdict[key]
            pfms.append(metric(output, target, *args, **kwargs))
        return pfms

    def reset(self):
        for val in self.mdict.values():
            val.reset()

    @property
    def results(self):
        res = {}
        for name, metric in self.mdict.items():
            res.update(metric.result)
        return res

    @property
    def objectives(self):
        return {name: self.mdict.get(name).objective for name in self.mdict}

    def __getattr__(self, key):
        return self.mdict[key]
```

### 5.5 自定义Trainer

可通过自定义callback的方式来自定义trainer，callback的实现可参考vega提供的缺省的callback。
如下是其中的ModelStatistics的实现：

```python
@ClassFactory.register(ClassType.CALLBACK)
class ModelStatistics(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        self.priority = 220

    def before_train(self, logs=None):
        self.input = None
        self.flops = None
        self.params = None
        self.calc_params_each_epoch = self.trainer.config.calc_params_each_epoch
        if vega.is_tf_backend():
            data_iter = self.trainer.valid_input_fn().make_one_shot_iterator()
            input_data, _ = data_iter.get_next()
            self.input = input_data[:1]

    def after_train_step(self, batch_index, logs=None):
        try:
            if self.input is None:
                input, target = logs['train_batch']
                self.input = torch.unsqueeze(input[0], 0)
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)

    def after_epoch(self, epoch, logs=None):
        if self.calc_params_each_epoch:
            self.update_flops_params(epoch=epoch, logs=logs)

    def after_train(self, logs=None):
        if not self.calc_params_each_epoch:
            self.update_flops_params(logs=logs)

    def update_flops_params(self, epoch=None, logs=None):
        self.model = self.trainer.model
        try:
            if self.flops is None:
                flops_count, params_count = calc_model_flops_params(self.model, self.input)
                self.flops, self.params = flops_count * 1e-9, params_count * 1e-3
            summary_perfs = logs.get('summary_perfs', {})
            if epoch:
                summary_perfs.update(
                    {'flops': self.flops, 'params': self.params, 'epoch': epoch})
            else:
                summary_perfs.update({'flops': self.flops, 'params': self.params})
            logs.update({'summary_perfs': summary_perfs})
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)
```

## 6. Configuration

Vega Configuration采用注册机制，所有注册的类都可以采用如下方法调用：

```python
_cls = ClassFactory.get_cls(class_type, class_name)
install = _cls(params)
```

同时Vega可以根据class type动态的映射配置文件中的配置到对应的实例上，从而使得开发者和用户能够直接使用`config`属性，无感知配置文件的加载和解析的过程。

比如如下是Prune-EA算法的NAS阶段的配置文件：

```yaml
nas:
    pipe_step:
        type: SearchPipeStep

    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
            train_portion: 0.9
        test:
            batch_size: 1024

    search_algorithm:
        type: PruneEA
        codec: PruneCodec
        policy:
            length: 464
            num_generation: 31
            num_individual: 32
            random_samples: 64

    search_space:
        type: SearchSpace
        modules: ['backbone']
        backbone:
            name: 'PruneResNet'
            base_chn: [16,16,16,32,32,32,64,64,64]
            base_chn_node: [16,16,32,64]
            num_classes: 10

    trainer:
        type: Trainer
        callbacks: PruneTrainerCallback
        epochs: 1
        init_model_file: "/cache/models/resnet20.pth"
        optim:
            type: SGD
            params:
                lr: 0.1
                momentum: 0.9
                weight_decay: !!float 1e-4
        lr_scheduler:
            type: StepLR
            params:
                step_size: 20
                gamma: 0.5
        seed: 10
        limits:
            flop_range: [!!float 0, !!float 1e10]
```

在trainer中，获取dataset和lr_scheduler的代码如下：

```python
    dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
    dataset = dataset_cls(mode=mode)
    search_alg_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM)
    search_alg = search_alg_cls(search_space)
```

从如上可以看出，并不需要指定类名称，vega通过扫描配置文件，确定当前所在的pipestep，通过类的类型，找到类名称和参数，供classFactory返回正确的类定义。
Vega支持的pipestep的具体定义可参考[配置参考](../user/config_reference.md)

## 7. pipeline

Vega的Pipeline通过加载`config`配置来实现多个`pipestep`的串联，用户执行`vega.run('config.yml')`的时候会先执行`_init_env(cfg_path)`方法加载配置，然后调用`Pipeline().run()`根据配置文件中的定义运行具体的`pipestep`的`do()`函数。
在config.yml中使用`pipleline`来定义`pipestep`的执行顺序，如下例中，`pipeline: [nas, fullytrain]`表示首先执行`nas`节点的`pipestep`，然后执行`fullytrain`节点的`pipestep`。

```yaml
pipeline: [nas, fullytrain]

nas:
    pipe_step:
        type: SearchPipeStep

fullytrain:
    pipe_step:
        type: TrainPipeStep
```

### 7.1 Report

一个Pipeline中包含了多个步骤，这些步骤之间的数据传递，可以通过Report来完成。Report实时收集各个Step的训练过程数据和评估结果，供本步骤和随后的步骤查询，同时Report的数据也会实时的保存到文件中。

Report提供的主要接口如下，在模型训练时，需要调用`update()`接口保存训练结果。搜索算法可调用`pareto_front()`接口来获取评估结果，这两个接口是最常用的接口。
Trainer已集成了Report的调动，在完成训练和评估后，Trainer会将结果数据自动的调用Report接口收集数据，供搜索算法使用。

```python
@singleton
class ReportServer(object):

    @property
    def all_records(self):

    def pareto_front(self, step_name=None, nums=None, records=None):

    @classmethod
    def close(cls, step_name, worker_id):


class ReportClient(object):

    @classmethod
    def get_record(cls, step_name, worker_id):

    @classmethod
    def update(cls, record):
```

### 7.2 扩展`pipestep`

当前已预置的`pipestep`有：

* SearchPipeStep
* TrainPipeStep
* BechmarkPipeStep

若需要扩展`pipestep`，需要继承基类`PipeStep`，实现`do()`函数即可，具体可参考如上类的实现代码：

```python
class PipeStep(object):

    def do(self):
        """Do the main task in this pipe step."""
        pass
```
