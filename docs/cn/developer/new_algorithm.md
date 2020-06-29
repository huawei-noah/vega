# 算法开发指导

向Vega库中新增算法，如新的网络搜索算法、模型压缩算法、超参优化算法、数据增广算法等，需要基于Vega提供的基础类进行扩展，并将新增的文件放到合适的目录。

## 1. 新增架构搜索或模型压缩算法

新增架构搜索算法和新增模型压缩算法是类似的，本例以模型压缩算法`Prune-EA`为例，来说明如何新增一个架构搜索算法或模型压缩算法到Vega算法库。

### 1.1 从配置文件开始

首先，我们从配置文件开始，Vega的任何算法都是通过配置文件中的配置项来配置，并在运行中加载这些配置信息，将算法的各个部件组合起来，形成一个完成的算法，并有配置文件来控制算法的运行流程。
对于这个新增的`Prune-EA`算法，配置文件如下：

```yaml
pipeline: [nas]

nas:
    pipe_step:
        type: NasPipeStep

    dataset:
        type: Cifar10

    search_algorithm:
        type: PruneEA
        codec: PruneCodec
        policy:
            length: 464
            num_generation: 31
            num_individual: 4
            random_models: 32

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
        epochs: 2
        init_model_file: "./bestmodel.pth"
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
        metrics:
            type: accuracy
        seed: 10
        limits:
            flop_range: [!!float 0, !!float 1e10]
```

在该的配置文件中：

1. 配置项`pipeline`是一个数组，包含了多个`Pipe Step`，Vega依次加载这些步骤，形成一个完整的运行流程。如本例中只有一个步骤`nas1`，这个名字可由用户自定义。
2. 配置项`nas1`下的配置项`pipe_step`的`type`属性是`Vega`库中提供的`NasPipeStep`，专门用于网络架构搜索和模型压缩。
3. 配置项`nas1`下的配合项`dataset`是该算法使用的数据集，当前配置的是`Vega`中的数据库类`Cifar10`。
4. 配置项`nas1`下的其他三个配置项是`search_space`、`search_algorithm`、`trainer`是四个和算法密切相关的配置项，这四个配置项对应的类分别是`PruneResNet`、`PruneEA`、`PruneTrainer`，本文随后会重点介绍这三部分。

### 1.2 设计搜索空间

我们假定这个模型压缩算法是一个剪枝算法，针对`ResNet`网络，尝试完全剪除该网络的一层或多层。我们假设Vega库中提供的搜索空间都不满足该算法要求，需要新增搜索空间。我们把这个搜索空间对应的网络命名为`PruneResNet`，该网络的关键属性是`base channel`和`base channel node`，那么该算法的搜索空间可定义为：

```yaml
    search_space:
        type: SearchSpace
        modules: ['backbone']
        backbone:
            name: 'PruneResNet'
            base_chn: [16,16,16,32,32,32,64,64,64]
            base_chn_node: [16,16,32,64]
            num_classes: 10
```

针对这个网络的初始化算法如下：

```python
@NetworkFactory.register(NetTypes.BACKBONE)
class PruneResNet(Network):

    def __init__(self, descript):
        super(PruneResNet, self).__init__()
        self.net_desc = descript
        block = descript.get('block', 'PruneBasicBlock')
        if block == 'PruneBasicBlock':
            self.block = eval(block)
        else:
            raise TypeError('Do not have this block type: {}'.format(block))
        self.encoding = descript.get('encoding')
        self.chn = descript.get('chn')
        self.chn_node = descript.get('chn_node')
        self.chn_mask = descript.get('chn_mask', None)
        self.chn_node_mask = descript.get('chn_node_mask', None)
        num_blocks = descript.get('num_blocks', [3, 3, 3])
        num_classes = descript.get('num_classes', 10)
        self.in_planes = self.chn_node[0]
        self.conv1 = nn.Conv2d(
            3, self.chn_node[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.chn_node[0])
        self.layer1 = self._make_layer(
            self.block, self.chn_node[1], self.chn[0:3], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            self.block, self.chn_node[2], self.chn[3:6], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            self.block, self.chn_node[3], self.chn[6:9], num_blocks[2], stride=2)
        self.linear = nn.Linear(self.chn_node[3], num_classes)
```

构造函数只接受一个`descript`参数，这个参数是一个`dict`类型，包含了构造`PruneResNet`所需要的所有参数，初始化函数需要对`descript`参数解析成具体值后生成网络结构。

`PruneResNet`还需注册到`NetworkFactory`中，并分类为`NetTypes.BACKBONE`，示例中`@NetworkFactory.register(NetTypes.BACKBONE)`完成了注册过程。

### 1.3 设计搜索算法

我们假设这个新的算法使用进化算法，同时Vega提供的算法都不满足要求，需要新实现进化算法`PruneEA`，适配`PruneResNet`，针对进化算法，我们需要提供进化代数，种群数量等，同时我们还需要定义进化编解码的函数，配置文件如下。

```yaml
    search_algorithm:
        type: PruneEA
        codec: PruneCodec
        policy:
            length: 464
            num_generation: 31
            num_individual: 4
            random_models: 32
```

该算法的初始化代码如下：

```python
@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class PruneEA(SearchAlgorithm):

    def __init__(self, search_space):
        super(PruneEA, self).__init__(search_space)
        self.length = self.policy.length
        self.num_individual = self.policy.num_individual
        self.num_generation = self.policy.num_generation
        self.x_axis = 'flops'
        self.y_axis = 'acc'
        self.random_models = self.policy.random_models
        self.codec = Codec(self.cfg.codec, search_space)
        self.random_count = 0
        self.ea_count = 0
        self.ea_epoch = 0
        self.step_path = FileOps.join_path(self.local_output_path, self.cfg.step_name)
        self.pd_file_name = FileOps.join_path(self.step_path, "performance.csv")
        self.pareto_front_file = FileOps.join_path(self.step_path, "pareto_front.csv")
        self.pd_path = FileOps.join_path(self.step_path, "pareto_front")
        FileOps.make_dir(self.pd_path)
```

同样，`PruneEA`类需要注册到`ClassFactory`中，以上代码中的`@ClassFactory.register(ClassType.SEARCH_ALGORITHM)`完成了注册过程。

PruneEA中最重要的是要实现`search()`的接口，负责从`search space`中搜索出一个网络描述。实现代码如下：

```python
    def search(self):
        if self.random_count < self.random_models:
            self.random_count += 1
            return self.random_count, self._random_sample()
        pareto_front_results = self.get_pareto_front()
        pareto_front = pareto_front_results["encoding"].tolist()
        if len(pareto_front) < 2:
            encoding1, encoding2 = pareto_front[0], pareto_front[0]
        else:
            encoding1, encoding2 = random.sample(pareto_front, 2)
        choice = random.randint(0, 1)
        # mutate
        if choice == 0:
            encoding1List = str2list(encoding1)
            encoding_new = self.mutatation(encoding1List)
        # crossover
        else:
            encoding1List = str2list(encoding1)
            encoding2List = str2list(encoding2)
            encoding_new, _ = self.crossover(encoding1List, encoding2List)
        self.ea_count += 1
        net_desc = self.codec.decode(encoding_new)
        return self.random_count + self.ea_count, net_desc
```

`search()`函数返回的是搜索的`id`和网络描述给`Generator`。

`PruneEA`搜索算法使用到了编码，所以需要使用一个编解码器`Codec`，负责将编码解析成网络描述。其编解码器`PruneCodec`的实现代码如下：

```python
class PruneCodec(Codec):

    def __init__(self, codec_name, search_space):
        super(PruneCodec, self).__init__(codec_name, search_space)
        self.search_space = search_space.search_space

    def decode(self, code):
        chn_info = self._code_to_chninfo(code)
        desc = {
            "backbone": chn_info
        }
        desc = update_dict(desc, copy.deepcopy(self.search_space))
        return NetworkDesc(desc)
```

`PruneCodec`中的`decode()`函数接受一个编码`code`参数，通过与实际网络描述的对应关系，解码成`PruneResNet`初始化参数的网络描述。

### 1.4 定制模型训练过程

在该剪枝算法中，训练过程某些步骤具有一定的独特性，比如在加载模型和保存模型上。在这两个接口上需要重写标准`Trainer`里面的函数。

在加载模型中，剪枝模型的预训练模型文件是剪枝前的模型，所以需要加载在剪枝前的模型上，再利用剪枝后模型的需要`mask`的位置进行剪枝操作。

在保存模型中，剪枝模型不只保存训练后的模型权重，还要保存模型的`flops`、参数大小、准确率和模型编码信息等。

相关代码可参考 [PruneTrainer](../../../vega/algorithms/compression/prune_ea/prune_trainer_callback.py)。

### 1.5 示例

该剪枝算法的完整实现可参数Vega SDK中`vega/algorithms/compression/prune_ea`目录下的代码。

```python
import vega


if __name__ == '__main__':
    vega.run('./prune.yml')
```

执行如下命令：

```bash
python main.py
```

运行结束后，会保存成一个文件，会在输出目录下保存模型描述文件和性能评估结果。

## 2. 新增超参优化算法

新增超参数优化算法可以参照当前已有的超参优化算法实现，进行增加。本例以新增超参优化算法`MyHpo`为例，来说明如何将其加入到Vega算法库。

由于当前的超参数搜索空间相对固定，Vega中的超参数优化算法要求可以方便的替换，即对于同一个搜索任务、搜索空间，要求可以直接替换另一个超参优化算法。所以这里的新增`MyHpo`的例子，我们直接使用`examples`中的`asha`的例子，并替换其算法为`MyHpo`。

### 2.1 配置文件

首先，我们从配置文件开始，Vega的任何算法都是通过配置文件中的配置项来配置，并在运行中加载这些配置信息，将算法的各个部件组合起来，形成一个完成的算法，并有配置文件来控制算法的运行流程。
对于这个新增的`MyHpo`算法，可以采用如下的与`asha`类似的配置文件：

```yaml
# 'myhpo.yml'
pipeline: [hpo1]

hpo1:
    pipe_step:
        type: HpoPipeStep
    dataset:
        type: Cifar10
    hpo:
        type: MyHpo
        total_epochs: 81
        config_count: 40
        hyperparameter_space:
            hyperparameters:
                - key: dataset.batch_size
                  type: INT_CAT
                  range: [8, 16, 32, 64, 128, 256]
                - key: trainer.optim.lr
                  type: FLOAT_EXP
                  range: [0.00001, 0.1]
                - key: trainer.optim.type
                  type: STRING
                  range: ['Adam', 'SGD']
                - key: trainer.optim.momentum
                  type: FLOAT
                  range: [0.0, 0.99]
            condition:
                - key: condition_for_sgd_momentum
                  child: trainer.optim.momentum
                  parent: trainer.optim.type
                  type: EQUAL
                  range: ["SGD"]

    trainer:
        type: Trainer

    model:
        model_desc:
            modules: ["backbone", "head"]
            backbone:
                base_channel: 64
                downsample: [1, 0, 1, 0, 0, 0, 1, 0]
                base_depth: 18
                doublechannel: [1, 0, 1, 0, 0, 0, 1, 0]
                name: ResNetVariant
            head:
                num_classes: 10
                name: LinearClassificationHead
                base_channel: 512

    evaluator:
        type: Evaluator
        gpu_evaluator:
            type: GpuEvaluator
```

在以上的配置文件中：

1. 配置项`pipeline`是一个数组，包含了多个`Pipe Step`，Vega依次加载这些步骤，形成一个完整的运行流程。如本例中只有一个步骤`hpo1`，这个名字可由用户自定义。
2. 配置项`hpo1`下的配置项`pipe_step`的`type`属性是`Vega`库中提供的`HpoPipeStep`，专门用于超参数优化，该Pipestep会利用Trainer训练模型，然后交给Evaluator来评估模型，得到模型的评估结果。
3. 配置项`hpo1`下的配合项`dataset`是该算法使用的数据集，当前配置的是`Vega`中的数据库类`Cifar10`。
4. 配置项`hpo1`下的其他三个配置项是`hpo`、`trainer`、`evaluator`。其中`hpo`是算法相关的配置项指定了使用type=MyHpo算法，超参数空间设定还是使用`asha`示例中的空间设定。`trainer`、`evaluator`分别对应当前模型训练和模型评估两部分的配置，这里我们使用默认的`Trainer`和`Evaluator`，模型选择了一个随机生成的ResNetVarient，且配置类似于`asha`示例的配置。可以看到整个配置项中与`asha`示例不同的唯一之处就是修改了`hpo`中的算法type，从`AshaHpo`改为了`MyHpo`，这里默认新增的算法没有更多特殊配置项，如果有可以在`hpo`中新增。

### 2.2 设计Generator

当前为了解耦核心算法和`vega.pipestep`，我们增加了一个中间层，即`HpoGenerator`，用于对接`pipestep`的接口，同时屏蔽底层不同优化算法的差异性。所以用户需要首先新增一个`MyHpo`类继承自vega提供的`vega.core.pipeline.hpo_generator.HpoGenerator`基类。

这个`MyHpo`类需要实现`proposal`、`is_completed`、`best_hps`、`update_performance`等接口，在实现这几个接口的时候，需要去调用用户自己实现的核心优化算法`MyAlg`中对应的`proposal`、`is_completed`、`update`等方法，而在`MyHpo`类主要用于解决两端的接口结合的问题，消除差异性。

```python
@ClassFactory.register(ClassType.HPO)
class MyHpo(HpoGenerator):
    def __init__(self):
        super(MyHpo, self).__init__()
        hps = json_to_hps(self.cfg.hyperparameter_space)
        self.hpo = MyAlg(hps, self.cfg.config_count,
                        self.cfg.total_epochs)
    def proposal(self):
        # 更多适配操作
        sample = self.hpo.propose()
        # 更多适配操作
        return sample

    @property
    def is_completed(self):
        return self.hpo.is_completed

    @property
    def best_hps(self):
        return self.hpo.best_config()

    def update_performance(self, hps, performance):
        # 更多适配操作
        self.hpo.add_score(hps, performance)
        # 更多适配操作

    def _save_score_board(self):
        pass
```

另外`HpoGenerator`基类中提供了部分保存hpo中间结果和最终结果的方法，用户也可以考虑实现`_save_score_board`等接口，用于在算法中输出算法运行的中间结果，具体方法可以参照vega中实现的类似算法的中间结果输出。

### 2.3 设计核心算法MyAlg

具体的超参数优化算法在`MyAlg`中来完成，我们提供了`hyperparameter_space`用于将当前超参数空间进行映射、对空间中超参之间进行条件约束、并提供从超参空间进行采样和反向映射等功能。

`MyAlg`可以按设计人员需要设计不同的接口，差异性的接口，可以通过上述`MyHpo`类进行封装屏蔽，从而最终保证暴露给`pipestep`的接口是统一的。

具体的`MyAlg`的设计，算法人员可以参考vega中当前已经实现的各种超参优化算法。

```python
class MyAlg(object):
    def __init__(self, hyperparameter_space, *args, **kargs):
        self.is_completed = False
        self.hyperparameter_space = hyperparameter_space
        parameters = self.hyperparameter_space.get_sample_space()

    def propose(self):
        raise NotImplementedError

    def add_score(self, config_id, rung_id, score):
        raise NotImplementedError

    def best_config(self):
        raise NotImplementedError
```

### 2.4 调测

下一步就是在安装了Vega的环境下进行调测，命令如下：

```bash
python main.py
```

成功运行后，在算法运行过程中，会在输出目录中输出`best_hps.json`保存当前最佳的超参数，以及`hps.csv`保存超参、id和评分的表格。
