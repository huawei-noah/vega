# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoPrune."""
import logging

from vega.algorithms.compression.auto_prune.pruning import named_pruned_modules, compress
from vega.algorithms.hpo.sha_base.tuner import TunerBuilder
from vega.common import ClassType, ClassFactory, ConfigSerializable, callbacks
from vega.core.pipeline.conf import PipeStepConfig
from vega.core.search_algs import SearchAlgorithm
from vega.core.search_space import SpaceSet
from vega.model_zoo import ModelZoo


class AutoPruneConfig(ConfigSerializable):
    """DAG Block Nas Config."""

    num_samples = 100
    tuner = 'GP'  # GP | RF | RandSearch | hebo
    objective_keys = 'accuracy'
    space_range = [50, 99]
    space_key = 'prune_d_rate'  # prune_d_rate | prune_rate
    each_rung_samples = num_samples
    strategy = None  # l1
    prune_type = 'prune'  # prune | mask
    hps_handler = None  # progressive


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class AutoPrune(SearchAlgorithm):
    """Auto Prune Base Class."""

    config = AutoPruneConfig()

    def __init__(self, search_space=None, model=None, **kwargs):
        super().__init__(search_space, **kwargs)
        self.model = model or ModelZoo().get_model(PipeStepConfig.model.model_desc,
                                                   PipeStepConfig.model.pretrained_model_file)
        self.search_space = self.encode(self.search_space)
        self.sample_count = 0
        if self.config.tuner == 'hebo':
            from vega.algorithms.hpo.sha_base.hebo_adaptor import HeboAdaptor
            self.tuner = HeboAdaptor(self.search_space)
        else:
            self.tuner = TunerBuilder(search_space=self.search_space, tuner=self.config.tuner)

    def encode(self, search_space, space_key=None, space_range=None):
        """Encode searchspace."""
        space_key = space_key or self.config.space_key
        space_range = space_range or self.config.space_range
        search_space = search_space or SpaceSet().load([(space_key, 'INT', space_range)])
        space_set = SpaceSet()
        item = search_space.get("hyperparameters")[0]
        for name, module in named_pruned_modules(self.model):
            space_set.add("{}.{}".format(name, item.get("key")), space_type=item.get("type"),
                          space_range=item.get("range"))
        # first conv not pruned.
        space_set.pop(0)
        return space_set.search_space

    @classmethod
    def decode(cls, sample):
        """Decode desc."""
        return sample.get("worker_id"), None, sample.get("encoded_desc"), dict(objective_keys=cls.config.objective_keys)

    def search(self):
        """Search a desc."""
        desc = self.do_search()
        self.sample_count += 1
        desc['trainer.epochs'] = self.sample_count // self.config.each_rung_samples - 1
        return dict(worker_id=self.sample_count - 1, encoded_desc=dict(desc))

    def do(self, show_desc=False):
        """Prune once."""
        desc = self.search().get("encoded_desc")
        model = compress(self.model, desc, self.config.strategy, self.config.prune_type)
        if show_desc:
            return model, desc
        return model

    def update(self, records):
        """Update records."""
        features = records.get('hps')
        if 'trainer.epochs' in features:
            features.pop("trainer.epochs")
        labels = records.get('rewards')
        labels = labels[0] if isinstance(labels, list) else labels
        self.tuner.add(features, labels)

    def do_search(self):
        """Search desc until match the expected ratio."""
        if self.config.hps_handler == 'progressive':
            self.search_space.handler = ProgressiveHandler(self.sample_count, self.config.num_samples)
        return self.tuner.propose()[0]

    @property
    def is_completed(self):
        """Check is completed."""
        return self.sample_count >= self.config.num_samples

    def get_best(self, show_desc=False):
        """Get best score and hps."""
        if show_desc:
            return self.tuner._best_score, self.tuner._best_hyperparams
        return self.tuner._best_score

    @staticmethod
    @callbacks("init_trainer")
    def prune(trainer, logs=None):
        """Define prune function to init trainer callback."""
        logging.info("model prune hps: {}".format(trainer.hps))
        trainer.model = compress(trainer.model, trainer.hps)


class ProgressiveHandler(object):
    """Define a Handler for search space."""

    def __init__(self, curr, max_samples):
        self.rate = curr / max_samples
        logging.debug("Progress search space rate: {}".format(self.rate))

    def __call__(self, low, high):
        """Call sample to change search space range."""
        low = high - (high - low) * self.rate
        return low, high
