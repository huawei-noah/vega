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

"""Implementation of Progressive Shrinking in Once for All."""
import itertools
import time
import random
from modnas.estim.base import EstimBase
from modnas.registry.estim import register
from modnas.contrib.arch_space.elastic.spatial import ElasticSpatial
from modnas.contrib.arch_space.elastic.sequential import ElasticSequential
from modnas import backend


@register
class ProgressiveShrinkingEstim(EstimBase):
    """Applies the Progressive Shrinking training strategy on a supernet."""

    def __init__(self,
                 *args,
                 stages,
                 use_ratio=False,
                 n_subnet_batch=1,
                 stage_rerank_spatial=True,
                 num_bn_batch=100,
                 clear_subnet_bn=True,
                 save_stage=False,
                 reset_stage_training=True,
                 subnet_valid_freq=25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.stages = stages
        self.n_subnet_batch = n_subnet_batch
        self.use_ratio = use_ratio
        self.save_stage = save_stage
        self.reset_stage_training = reset_stage_training
        self.spatial_candidates = None
        self.sequential_candidates = None
        self.subnet_results = dict()
        self.cur_stage = -1
        self.stage_rerank_spatial = stage_rerank_spatial
        self.num_bn_batch = num_bn_batch
        self.clear_subnet_bn = clear_subnet_bn
        self.subnet_valid_freq = subnet_valid_freq

    def set_stage(self, stage):
        """Set PS training stage from config."""
        self.set_spatial_candidates(stage.get('spatial', None))
        self.set_sequential_candidates(stage.get('sequential', None))

    def set_sequential_candidates(self, candidates):
        """Set PS sequential (depth) candidates from config."""
        n_groups = ElasticSequential.num_groups()
        if n_groups == 0 or candidates is None:
            candidates = [[None]]
        elif not isinstance(candidates[0], (list, tuple)):
            candidates = [candidates] * n_groups
        self.sequential_candidates = candidates
        self.logger.info('set sequential candidates: {}'.format(self.sequential_candidates))

    def set_spatial_candidates(self, candidates):
        """Set PS spatial (width) candidates from config."""
        n_groups = ElasticSpatial.num_groups()
        if n_groups == 0 or candidates is None:
            candidates = [[None]]
        elif not isinstance(candidates[0], (list, tuple)):
            candidates = [candidates] * n_groups
        self.spatial_candidates = candidates
        self.logger.info('set spatial candidates: {}'.format(self.spatial_candidates))

    def randomize(self, seed=None):
        """Randomize sampling."""
        if seed is None:
            seed = time.time()
        random.seed(seed)

    def apply_subnet_config(self, config):
        """Apply sampled config to obtain the subnet."""
        self.logger.debug('set subnet: {}'.format(config))
        spatial_config = config.get('spatial', None)
        for i, sp_g in enumerate(ElasticSpatial.groups()):
            if spatial_config is None or len(spatial_config) <= i:
                width = None
            else:
                width = spatial_config[i]
            if self.use_ratio:
                sp_g.set_width_ratio(width)
            else:
                sp_g.set_width(width)
        sequential_config = config.get('sequential', None)
        for i, sp_g in enumerate(ElasticSequential.groups()):
            if sequential_config is None or len(sequential_config) <= i:
                depth = None
            else:
                depth = sequential_config[i]
            if self.use_ratio:
                sp_g.set_depth_ratio(depth)
            else:
                sp_g.set_depth(depth)

    def sample_spatial_config(self, seed=None):
        """Sample spatial config from candidates."""
        self.randomize(seed)
        spatial_config = []
        for sp_cand in self.spatial_candidates:
            width = random.choice(sp_cand)
            spatial_config.append(width)
        return {
            'spatial': spatial_config,
        }

    def sample_sequential_config(self, seed=None):
        """Sample sequential config from candidates."""
        self.randomize(seed)
        sequential_config = []
        for sq_cand in self.sequential_candidates:
            depth = random.choice(sq_cand)
            sequential_config.append(depth)
        return {
            'sequential': sequential_config,
        }

    def sample_config(self, seed=None):
        """Sample config (spatial & sequential) from candidates."""
        config = dict()
        if self.spatial_candidates is not None:
            config.update(self.sample_spatial_config(seed=seed))
        if self.sequential_candidates is not None:
            config.update(self.sample_sequential_config(seed=seed))
        return config

    def loss(self, data, output=None, model=None, mode=None):
        """Compute loss from subnet(s)."""
        model = self.model if model is None else model
        output = None
        if mode == 'train':
            visited = set()
            loss = None
            for _ in range(self.n_subnet_batch):
                config = self.sample_config(seed=None)
                key = str(config)
                if key in visited:
                    continue
                if loss is not None:
                    loss.backward()
                self.apply_subnet_config(config)
                loss = super().loss(data, output, model, mode)
                visited.add(key)
        else:
            loss = super().loss(data, output, model, mode)
        return loss

    def train_stage(self):
        """Train and evaluate supernet with current stage config."""
        config = self.config
        tot_epochs = config.epochs
        if self.reset_stage_training:
            self.reset_trainer()
        for epoch in itertools.count(self.cur_epoch + 1):
            if epoch == tot_epochs:
                break
            # train
            self.train_epoch(epoch, tot_epochs)
            # valid subnets
            if self.subnet_valid_freq != 0 and (epoch + 1) % self.subnet_valid_freq == 0:
                results = self.valid_subnet(epoch, tot_epochs)
                for name, res in results.items():
                    self.logger.info('Subnet {}: {:.4%}'.format(name, res))
                self.update_results(results)

    def update_results(self, results):
        """Merge subnet evaluation results."""
        for k, v in results.items():
            val = self.subnet_results.get(k, 0)
            self.subnet_results[k] = max(val, v)

    def state_dict(self):
        """Save training stage progress."""
        return {'cur_stage': self.cur_stage}

    def load_state_dict(self, state_dict):
        """Resume training stage progress."""
        if 'cur_stage' in state_dict:
            self.cur_stage = state_dict['cur_stage']

    def run(self, optim):
        """Train supernet in multiple PS stages."""
        self.reset_trainer()
        for self.cur_stage in itertools.count(self.cur_stage + 1):
            if self.cur_stage >= len(self.stages):
                break
            self.logger.info('running stage {}'.format(self.cur_stage))
            stage = self.stages[self.cur_stage]
            self.set_stage(stage)
            if self.stage_rerank_spatial:
                self.rerank_spatial()
            self.train_stage()
            if self.save_stage:
                self.save_checkpoint(-1, 'stage_{}'.format(self.cur_stage))
        results = {
            'best_top1': None if not self.subnet_results else max([acc for acc in self.subnet_results.values()]),
            'subnet_best_top1': self.subnet_results,
        }
        return results

    def rerank_spatial(self):
        """Reorder channel dimensions in all spatial groups."""
        for g in ElasticSpatial.groups():
            g.set_spatial_rank()

    def valid_subnet(self, *args, configs=None, **kwargs):
        """Sample and validate subnets from current candidates."""
        if configs is None:
            configs = dict()
            sp_len = ElasticSpatial.num_groups()
            sp_cand = self.spatial_candidates[0]
            sp_dim = len(sp_cand)
            sq_len = ElasticSequential.num_groups()
            sq_cand = self.sequential_candidates[0]
            sq_dim = len(sq_cand)
            for sp_idx, sq_idx in itertools.product(range(sp_dim), range(sq_dim)):
                sp_val = sp_cand[sp_idx]
                sq_val = sq_cand[sq_idx]
                sp_config = [sp_val] * sp_len
                sq_config = [sq_val] * sq_len
                conf = {
                    'spatial': sp_config,
                    'sequential': sq_config,
                }
                name = 'sp_{}_sq_{}'.format(sp_val, sq_val)
                configs[name] = conf
        results = dict()
        for name, conf in configs.items():
            self.apply_subnet_config(conf)
            backend.recompute_bn_running_statistics(self.model, self.trainer, self.num_bn_batch, self.clear_subnet_bn)
            score = self.get_score(self.compute_metrics())
            results[name] = score
        return results
