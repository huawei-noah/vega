# Algorithm Development Guide

Add new algorithms to the Vega library, such as the new network search algorithm, model compression algorithm, hyperparameter optimization algorithm, and data augmentation algorithm. The algorithms need to be extended based on the basic class provided by Vega and the new files need to be stored in an appropriate directory.

## 1. Added architecture search or model compression algorithm

The procedure for adding an architecture search algorithm is similar to that for adding a model compression algorithm. The following uses the compression algorithm Prune-EA model as an example to describe how to add an architecture search algorithm or model compression algorithm to the Vega algorithm library.

### 1.1 Starting from the configuration file

First, we start from the configuration file. Any Vega algorithm is configured through the configuration items in the configuration file, and the configuration information is loaded during the running. The components of the algorithm are combined to form a complete algorithm, and the configuration file controls the running process of the algorithm. For the new Prune-EA algorithm, the following configuration file can be used:

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

In the preceding configuration file:

1. The pipeline configuration item is an array that contains multiple pipeline steps. The Vega loads these steps in sequence to form a complete running process. In this example, there is only one step, nas1. You can customize the name.
2. The type attribute of the pipe_step configuration item under the configuration item nas1 is NasPipeStep provided by the Vega library, which is dedicated to network architecture search and model compression.
3. The dataset under the configuration item nas1 is the data set used by the algorithm. Currently, the database class Cifar10 in Vega is configured.
4. The other three configuration items under the configuration item nas1 are search_space, search_algorithm, and trainer, which are closely related to the algorithm. The four configuration items correspond to the classes PruneResNet, PruneEA, and PruneTrainer respectively, this article will then focus on these three parts.

### 1.2 Design the search space.

Assume that the compression algorithm model is a pruning algorithm. For a ResNet network, one or more layers of the network are pruned. Assume that the search space provided by the Vega library does not meet the requirements of the algorithm. You need to add a search space. The network corresponding to the search space is named PruneResNet. The key attributes of the network are base channel and base channel node. The search space of the algorithm can be defined as follows:

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

The initialization algorithm for the network is as follows:

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

The constructor takes only one descriptor parameter of the dict type, which contains all parameters required for constructing PruneResNet. The initializer needs to parse the descriptor parameter to a specific value to generate the network structure.

PruneResNet also needs to be registered with NetworkFactory and classified as NetTypes.BACKBONE. In the example, the @NetworkFactory.register(NetTypes.BACKBONE) completes the registration process.

### 1.3 Designing a Search Algorithm

Assume that the new algorithm uses the evolutionary algorithm and the algorithms provided by Vega do not meet the requirements. The new evolutionary algorithm PruneEA needs to be implemented to adapt to PruneResNet. For the evolutionary algorithm, we need to provide the evolutionary algebra and population quantity. In addition, we need to define the function of evolutionary encoding and decoding, the configuration file is as follows:

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

The initialization code of the algorithm is as follows:

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

Similarly, the PruneEA class needs to be registered with the ClassFactory. The @ClassFactory.register(ClassType.SEARCH_ALGORITHM) in the preceding code completes the registration process.

The most important function of PruneEA is to implement the search() interface, which is used to search for a network description from the search space. The implementation code is as follows:

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

The search() function returns the search id and network description to the generator.

The PruneEA search algorithm uses codes. Therefore, a codec is required to parse the codes into network descriptions. The implementation code of the PruneCodec is as follows:

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

The decode() function in the PruneCodec accepts an encoding code parameter and decodes it into the network description of the PruneResNet initialization parameter based on the mapping with the actual network description.

### 1.4 Customized Model Training Process

In the pruning algorithm, some steps in the training process are unique, for example, loading and saving the model. The functions in the standard Trainer need to be rewritten on the two interfaces.

In the loading model, the pre-trained model file of the pruning model is the model before pruning. Therefore, the model needs to be loaded to the model before pruning, and then pruning is performed based on the location that needs to be masked of the pruned model.

When saving a model, a pruned model not only stores a trained model weight, but also stores model flaps, a parameter size, accuracy, model coding information, and the like of the model.

Refer to [prunetrainer](../../../vega/algorithms/compression/prune_ea/prune_trainer_callback.py) for relevant codes.

### 1.5 Example

The complete implementation of the pruning algorithm can be implemented by using the code in the vega/algorithms/compression/prune_ea directory of the Vega SDK parameter.

```python
import vega


if __name__ == '__main__':
    vega.run('./prune.yml')
```

Execute the following command:

```bash
python main.py
```

After running, it will be saved as a file, and the model description file and performance evaluation results will be saved in the output directory.

## 2. Add a new hyperparameter optimization algorithm

The new hyper-parameter optimization algorithm can be implemented by referring to the existing hyperparameter optimization algorithm. This example describes how to add the hyperparameter optimization algorithm MyHpo to the Vega algorithm library.

Because a current hyperparameter search space is relatively fixed, a hyperparameter optimization algorithm in Vega requires that the hyperparameter optimization algorithm can be conveniently replaced, that is, for a same search task and a same search space, another hyperparameter optimization algorithm can be directly replaced. Therefore, the example of adding MyHpo is directly used as the example of asha in examples, and the algorithm is replaced with MyHpo.

### 2.1 Configuration Files

First, we start from the configuration file. Any Vega algorithm is configured through the configuration items in the configuration file, and the configuration information is loaded during the running. The components of the algorithm are combined to form a complete algorithm, and the configuration file controls the running process of the algorithm. For the new MyHpo algorithm, the following configuration file similar to the asha file can be used:

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

In the preceding configuration file:

1. The pipeline configuration item is an array that contains multiple pipeline steps. The Vega loads these steps in sequence to form a complete running process. In this example, there is only one step, hpo1. You can customize the name.
2. The type attribute of the pipe_step configuration item in the hpo1 configuration item is HpoPipeStep provided by the Vega library, which is used for hyperparameter optimization. The PipeStep uses the Trainer to train the model and sends the model to the Evaluator for model evaluation.
3. The dataset under the configuration item hpo1 is the data set used by the algorithm. Currently, the database class Cifar10 in Vega is configured.
4. The other three configuration items under hpo1 are hpo, trainer, and evaluator. The hpo configuration item specifies whether to use the type=MyHpo algorithm, hyper-parameter space setting, or space setting in the asha example. Trainer and evaluator correspond to the configurations of the current model training and model evaluation, respectively. Here, the default Trainer and Evaluator are used. A randomly generated ResNetVarient is selected for the model, and the configuration is similar to that of the ASA example. The only difference between the configuration item and the ASHA example is that the algorithm type in the HPO is changed from AshaHpo to MyHpo. By default, no more special configuration items are added for the new algorithm. If there are special configuration items, add them to the HPO.

### 2.2 Design Generator

To decouple the core algorithm and vega.pipestep, an intermediate layer, that is, HpoGenerator, is added to interconnect with the pipestep interface and shield the differences between bottom-layer optimization algorithms. Therefore, you need to add a MyHpo class that inherits the vega.core.pipeline.hpo_generator.HpoGenerator base class provided by VEGA.

The MyHpo class needs to implement interfaces such as proposal, is_completed, best_hps, and update_performance. When implementing these interfaces, the class needs to invoke the corresponding methods in the core optimization algorithm MyAlg implemented by the user, such as proposal, is_completed, and update, the MyHpo class is mainly used to solve the problem of interface combination at both ends and eliminate differences.

```python
@ClassFactory.register(ClassType.HPO)
class MyHpo(HpoGenerator):
    def __init__(self):
        super(MyHpo, self).__init__()
        hps = json_to_hps(self.cfg.hyperparameter_space)
        self.hpo = MyAlg(hps, self.cfg.config_count,
                        self.cfg.total_epochs)
    def proposal(self):
        # more adaptation operations
        sample = self.hpo.propose()
        # more adaptation operations
        return sample

    @property
    def is_completed(self):
        return self.hpo.is_completed

    @property
    def best_hps(self):
        return self.hpo.best_config()

    def update_performance(self, hps, performance):
        # more adaptation operations
        self.hpo.add_score(hps, performance)
        # more adaptation operations

    def _save_score_board(self):
        pass
```

In addition, the HpoGenerator base class provides some methods for saving the HPO intermediate result and final result. You can implement interfaces such as _save_score_board to output the intermediate result of algorithm running. For details, see the intermediate result output of similar algorithms implemented in VEGA.

### 2.3 Designing MyAlg

The specific hyperparameter optimization algorithm is implemented in MyAlg. The domain_space is provided to map the current hyperparameter space, restrict the hyperparameters in the space, and provide the sampling and reverse mapping functions from the hyperparameter space.

MyAlg can design different interfaces based on the design personnel's requirements. Different interfaces can be encapsulated and shielded by using the MyHpo class to ensure that the interfaces exposed to the Pipestep are unified.

For details about the MyAlg design, refer to the hyperparameter optimization algorithms that have been implemented in VEGA.

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

### 2.4 Commissioning

The next step is to perform the commissioning in the environment where Vega is installed. The command is as follows:

```bash
python main.py
```

After the algorithm is successfully executed, the best_hps.json file is generated in the output directory to save the optimal hyperparameters, and the hps.csv file is used to save the hyperparameter, ID, and score tables.
