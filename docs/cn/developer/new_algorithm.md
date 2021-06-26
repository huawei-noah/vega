# 算法开发指导

向Vega库中新增算法，如新的网络搜索算法、模型压缩算法、超参优化算法、数据增广算法等，需要基于Vega提供的基础类进行扩展。AutoML算法的核心的核心是搜索空间、搜索算法、网络构造和评估，新增算法主要考虑这几个方面。

## 1. 新增架构搜索算法

本例以模型压缩算法`BackboneNAS`为例，来说明如何新增一个架构搜索算法到Vega算法库。

### 1.1 从配置文件开始

首先，我们从配置文件开始，Vega的任何算法都是通过配置文件中的配置项来配置，并在运行中加载这些配置信息，将算法的各个部件组合起来，形成一个完成的算法，并有配置文件来控制算法的运行流程。
对于这个新增的`BackboneNas`算法，配置文件如下：

```yaml
nas:
    pipe_step:
        type: SearchPipeStep

    search_algorithm:
        type: BackboneNas
        codec: BackboneNasCodec
        policy:
            num_mutate: 10
            random_ratio: 0.2
        range:
            max_sample: 100
            min_sample: 10

    search_space:
        hyperparameters:
            -   key: network.backbone.depth
                type: CATEGORY
                range: [18, 34, 50, 101]
            -   key: network.backbone.base_channel
                type: CATEGORY
                range:  [32, 48, 56, 64]
            -   key: network.backbone.doublechannel
                type: CATEGORY
                range: [3, 4]
            -   key: network.backbone.downsample
                type: CATEGORY
                range: [3, 4]
    model:
        model_desc:
            modules: ['backbone']
            backbone:
                type: ResNet

    trainer:
        type: Trainer
        epochs: 1
        loss:
            type: CrossEntropyLoss


    dataset:
        type: Cifar10
        common:
            data_path: /cache/datasets/cifar10/
```

在该的配置文件中，主要是search_algorithm、search_space、model三个章节，分别是搜索算法定义、搜素空间定义和网络模型定义三部分，以下会详细说明这三部分。

### 1.2 设计搜索空间和网络结构

我们希望针对ResNet网络，搜索空间定义为depth、base_channel，及doublechannel和downsample的位置，我们可以定义搜索空间和网络结构如下：

```yaml
    search_space:
        hyperparameters:
            -   key: network.backbone.depth
                type: CATEGORY
                range: [18, 34, 50, 101]
            -   key: network.backbone.base_channel
                type: CATEGORY
                range:  [32, 48, 56, 64]
            -   key: network.backbone.doublechannel
                type: CATEGORY
                range: [3, 4]
            -   key: network.backbone.downsample
                type: CATEGORY
                range: [3, 4]
    model:
        model_desc:
            modules: ['backbone']
            backbone:
                type: ResNet
```

针对这个网络的初始化算法如下：

```python
@ClassFactory.register(ClassType.NETWORK)
class ResNet(Module):
    """Create ResNet SearchSpace."""

    def __init__(self, depth=18, base_channel=64, out_plane=None, stage=4, num_class=10, small_input=True,
                 doublechannel=None, downsample=None):
        """Create layers.

        :param num_reps: number of layers
        :type num_reqs: int
        :param items: channel and stride of every layer
        :type items: dict
        :param num_class: number of class
        :type num_class: int
        """
        super(ResNet, self).__init__()
        self.backbone = ResNetGeneral(small_input, base_channel, depth, stage, doublechannel, downsample)
        self.adaptiveAvgPool2d = AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        out_plane = out_plane or self.backbone.output_channel
        self.head = Linear(in_features=out_plane, out_features=num_class)
```

网络的构造函数中接受depth、base_channel、doublechannel、downsample等参数，这些参数通过搜索算法传递过来。具体的ResNetGeneral实现，可参考<https://github.com/huawei-noah/vega/blob/master/vega/networks/resnet_general.py>。

### 1.3 设计搜索算法

BackboneNas算法使用采用进化算法，对此我们需要定义进化算法的编解码函数。

配置文件如下：

```yaml
    search_algorithm:
        type: BackboneNas
        codec: BackboneNasCodec
        policy:
            num_mutate: 10
            random_ratio: 0.2
        range:
            max_sample: 3 #100
            min_sample: 1 #10

    search_space:
        hyperparameters:
            -   key: network.backbone.depth
                type: CATEGORY
                range: [18, 34, 50, 101]
            -   key: network.backbone.base_channel
                type: CATEGORY
                range:  [32, 48, 56, 64]
            -   key: network.backbone.doublechannel
                type: CATEGORY
                range: [3, 4]
            -   key: network.backbone.downsample
                type: CATEGORY
                range: [3, 4]
```

搜索算法代码如下：

```python
@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BackboneNas(SearchAlgorithm):
    """BackboneNas.

    :param search_space: input search_space
    :type: SeachSpace
    """

    config = BackboneNasConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNas."""
        super(BackboneNas, self).__init__(search_space, **kwargs)
        # ea or random
        self.num_mutate = self.config.policy.num_mutate
        self.random_ratio = self.config.policy.random_ratio
        self.max_sample = self.config.range.max_sample
        self.min_sample = self.config.range.min_sample
        self.sample_count = 0
        logging.info("inited BackboneNas")
        self.pareto_front = ParetoFront(
            self.config.pareto.object_count, self.config.pareto.max_object_ids)
        self._best_desc_file = 'nas_model_desc.json'

    @property
    def is_completed(self):
        """Check if NAS is finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search in search_space and return a sample."""
        sample = {}
        while sample is None or 'code' not in sample:
            pareto_dict = self.pareto_front.get_pareto_front()
            pareto_list = list(pareto_dict.values())
            if self.pareto_front.size < self.min_sample or random.random() < self.random_ratio or len(
                    pareto_list) == 0:
                sample_desc = self.search_space.sample()
                sample = self.codec.encode(sample_desc)
            else:
                sample = pareto_list[0]
            if sample is not None and 'code' in sample:
                code = sample['code']
                code = self.ea_sample(code)
                sample['code'] = code
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample):
                sample = None
        self.sample_count += 1
        logging.info(sample)
        sample_desc = self.codec.decode(sample)
        print(sample_desc)
        return dict(worker_id=self.sample_count, encoded_desc=sample_desc)

    def random_sample(self):
        """Random sample from search_space."""
        sample_desc = self.search_space.sample()
        sample = self.codec.encode(sample_desc, is_random=True)
        return sample

    def ea_sample(self, code):
        """Use EA op to change a arch code.

        :param code: list of code for arch
        :type code: list
        :return: changed code
        :rtype: list
        """
        new_arch = code.copy()
        self._insert(new_arch)
        self._remove(new_arch)
        self._swap(new_arch[0], self.num_mutate // 2)
        self._swap(new_arch[1], self.num_mutate // 2)
        return new_arch

    def update(self, record):
        """Use train and evaluate result to update algorithm.

        :param performance: performance value from trainer or evaluator
        """
        perf = record.get("rewards")
        worker_id = record.get("worker_id")
        logging.info("update performance={}".format(perf))
        self.pareto_front.add_pareto_score(worker_id, perf)

    def _insert(self, arch):
        """Random insert to arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        idx = np.random.randint(low=0, high=len(arch[0]))
        arch[0].insert(idx, 1)
        idx = np.random.randint(low=0, high=len(arch[1]))
        arch[1].insert(idx, 1)
        return arch

    def _remove(self, arch):
        """Random remove one from arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        # random pop arch[0]
        ones_index = [i for i, char in enumerate(arch[0]) if char == 1]
        idx = random.choice(ones_index)
        arch[0].pop(idx)
        # random pop arch[1]
        ones_index = [i for i, char in enumerate(arch[1]) if char == 1]
        idx = random.choice(ones_index)
        arch[1].pop(idx)
        return arch

    def _swap(self, arch, R):
        """Random swap one in arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        while True:
            not_ones_index = [i for i, char in enumerate(arch) if char != 1]
            idx = random.choice(not_ones_index)
            r = random.randint(1, R)
            direction = -r if random.random() > 0.5 else r
            try:
                arch[idx], arch[idx + direction] = arch[idx + direction], arch[
                    idx]
                break
            except Exception:
                continue
        return arch

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
```

编解码的代码如下：

```python

@ClassFactory.register(ClassType.CODEC)
class BackboneNasCodec(Codec):
    """BackboneNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace

    """

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNasCodec."""
        super(BackboneNasCodec, self).__init__(search_space, **kwargs)

    def encode(self, sample_desc, is_random=False):
        """Encode.

        :param sample_desc: a sample desc to encode.
        :type sample_desc: dict
        :param is_random: if use random to encode, default is False.
        :type is_random: bool
        :return: an encoded sample.
        :rtype: dict

        """
        layer_to_block = {18: (8, [0, 0, 1, 0, 1, 0, 1, 0]),
                          34: (16, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
                          50: (16, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
                          101: (33, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])}
        default_count = 3
        base_depth = sample_desc['network.backbone.depth']
        double_channel = sample_desc['network.backbone.doublechannel']
        down_sample = sample_desc['network.backbone.downsample']
        if double_channel != down_sample:
            return None
        code = [[], []]
        if base_depth in layer_to_block:
            if is_random or double_channel != default_count:
                rand_index = random.sample(
                    range(0, layer_to_block[base_depth][0]), double_channel)
                code[0] = [0] * layer_to_block[base_depth][0]
                for i in rand_index:
                    code[0][i] = 1
            else:
                code[0] = copy.deepcopy(layer_to_block[base_depth][1])
            if is_random or down_sample != default_count:
                rand_index = random.sample(
                    range(0, layer_to_block[base_depth][0]), down_sample)
                code[1] = [0] * layer_to_block[base_depth][0]
                for i in rand_index:
                    code[1][i] = 1
            else:
                code[1] = copy.deepcopy(layer_to_block[base_depth][1])
        sample = copy.deepcopy(sample_desc)
        sample['code'] = code
        return sample

    def decode(self, sample):
        """Decode.

        :param sample: input sample to decode.
        :type sample: dict
        :return: return a decoded sample desc.
        :rtype: dict

        """
        if 'code' not in sample:
            raise ValueError('No code to decode in sample:{}'.format(sample))
        code = sample.pop('code')
        desc = copy.deepcopy(sample)
        if "network.backbone.doublechannel" in desc:
            desc["network.backbone.doublechannel"] = code[0]
        if "network.backbone.downsample" in desc:
            desc["network.backbone.downsample"] = code[1]
        if len(desc["network.backbone.downsample"]) != len(desc["network.backbone.doublechannel"]):
            return None
        logging.info("decode:{}".format(desc))
        return desc
```

### 1.4 示例

该剪枝算法的完整实现可参数Vega SDK中<https://github.com/huawei-noah/vega/tree/master/vega/algorithms/nas/backbone_nas>目录下的代码。

## 2. 新增超参优化算法

新增超参优化算法和NAS算法类似，可参考<https://github.com/huawei-noah/vega/tree/master/vega/algorithms/hpo>下的各个算法。
