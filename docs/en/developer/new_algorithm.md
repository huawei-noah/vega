# Algorithm Development Guide

New algorithms, such as new network search algorithms, model compression algorithm, hyperparameter optimization algorithms, and data augmentation algorithms, need to be extended based on the basic classes provided by Vega. The core of the AutoML algorithm is search space, search algorithm, network construction and evaluation. The new algorithm mainly considers these aspects.

## 1. Add a schema search algorithm

The compression algorithm `BackboneNAS` model is used as an example to describe how to add an architecture search algorithm to the Vega algorithm library.

### 1.1 Starting from the configuration file

First, let's start from the configuration file. Any algorithm of Vega is configured through configuration items in the configuration file and loaded during running. All components of the algorithm are combined to form a complete algorithm. A configuration file is used to control the algorithm running process.

For the new BackboneNas algorithm, the configuration file is as follows:

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

In the configuration file, the search_algorithm, search_space, and model sections define the search algorithm, search space, and network model respectively. The following sections describe the three sections in detail.

### 1.2 Design Search Space and Network Architecture

For the ResNet network, the search space is defined as depth, base_channel, doublechannel, and downsample. The search space and network structure can be defined as follows:

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

The initialization algorithm for this network is as follows:

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

The constructor of the network accepts parameters such as depth, base_channel, doublechannel, and downsample. These parameters are passed through the search algorithm. For details about the implementation of ResNetGeneral, see <https://github.com/huawei-noah/vega/blob/master/vega/networks/resnet_general.py>.

### 1.3 Designing a Search Algorithm

The BackboneNas algorithm uses an evolutionary algorithm. Therefore, we need to define the encoding and decoding functions of the evolutionary algorithm.

The configuration file is as follows:

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

The search algorithm code is as follows:

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

The encoding and decoding codes are as follows:

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

### 1.4 Example

The complete implementation of the pruning algorithm can be specified by the code in the <https://github.com/huawei-noah/vega/tree/master/vega/algorithms/nas/backbone_nas> directory of the Vega SDK.

## 2. The hyperparameter optimization algorithm is added

The new hyperparameter optimization algorithm is similar to the NAS algorithm. For details, see the algorithms in <https://github.com/huawei-noah/vega/tree/master/vega/algorithms/hpo>.
