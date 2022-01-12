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
"""This is DAG Cell for network."""
import copy
import logging
import random
from collections import OrderedDict

from vega.common import ClassFactory, ClassType, ConfigSerializable
from vega.core.search_algs import SearchAlgorithm

from vega.core.pipeline.conf import PipeStepConfig
from vega.model_zoo import ModelZoo
from vega.algorithms.nas.dag_block_nas.match_blocks import match_blocks, SpaceIterRecord, SpaceIter, \
    mutate_sub_blocks, check_latency
from vega.report.report_server import ReportServer


class DAGBlockNasConfig(ConfigSerializable):
    """DAG Block Nas Config."""

    num_samples = 100
    mutation_method = 'progressive'  # random/progressive
    check_latency = False


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DAGBlockNas(SearchAlgorithm):
    """DAG Block Nas."""

    config = DAGBlockNasConfig()

    def __init__(self, search_space=None):
        """Construct the AdelaideMutate class.

        :param search_space: Config of the search space
        """
        super(DAGBlockNas, self).__init__(search_space)
        self.search_space = search_space
        self.sample_count = 0
        self.sampled_desc = []
        model_config = PipeStepConfig.model
        self.org_model = ModelZoo().get_model(model_config.model_desc, model_config.pretrained_model_file)

    def search(self):
        """Search block desc."""
        desc = self._do_search()
        if not desc:
            return None
        hash_code = hash(str(desc))
        if hash_code in self.sampled_desc:
            return None
        self.sampled_desc.append(hash_code)
        self.sample_count += 1
        # save records.
        SpaceIterRecord().dump(self.get_local_worker_path(worker_id=self.sample_count))
        return dict(worker_id=self.sample_count - 1, encoded_desc=dict(desc))

    def _do_search(self):
        if self.config.mutation_method == 'random':
            return dict(RandomMutation(self.search_space, self.org_model).run())
        # get last best model and weights.
        records = ReportServer().get_pareto_front_records(choice=True)
        if records:
            model_desc = records[0].desc
            pretrained_model_file = records[0].weights_file
        else:
            model_desc = PipeStepConfig.model.model_desc
            pretrained_model_file = PipeStepConfig.model.pretrained_model_file
        self.org_model = ModelZoo().get_model(model_desc, pretrained_model_file)
        desc = dict(ProgressiveMutation(self.search_space, self.org_model).run())
        if self.config.check_latency:
            target_model = ModelZoo().get_model(desc)
            if not check_latency(target_model):
                return None
        return desc

    @property
    def is_completed(self):
        """Check is completed."""
        return self.sample_count >= self.config.num_samples

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.config.num_samples


class RandomMutation(object):
    """Do mutation with random sample."""

    def __init__(self, search_space, model):
        self.fused_blocks_nums_iter = SpaceIter(search_space, 'fused_blocks_nums')
        self.block_type_iter = SpaceIter(search_space, 'block_type')
        self.model = model

    def run(self):
        """Run random mutation."""
        return self._do_mutation()

    def _do_mutation(self, fused_blocks_radio=1, mutated_blocks_radio=1):
        """Run block mutate."""
        blocks = match_blocks(self.model)
        target_desc = OrderedDict(copy.deepcopy(self.model.to_desc()))
        blocks = self.fuse_sub_blocks(blocks, self.fused_blocks_nums_iter, fused_blocks_radio)
        blocks.pop(0)
        for block in blocks:
            if random.uniform(0, 1) > mutated_blocks_radio:
                continue
            target_desc = mutate_sub_blocks(block, target_desc, self.block_type_iter)
        return target_desc

    def fuse_sub_blocks(self, org_blocks, fused_blocks_nums_iter, fused_blocks_radio=1):
        """Fuse sub block."""
        first_block = True
        fused_blocks = []
        blocks = copy.deepcopy(org_blocks)
        while blocks:
            fuse_deep = next(fused_blocks_nums_iter)  # random.choice(fused_blocks_nums)
            need_fuse_blocks = []
            if first_block or len(blocks) <= 2 or random.uniform(0, 1) > fused_blocks_radio:
                fused_blocks.append(blocks.pop(0))
                first_block = False
                continue
            counts = fuse_deep if fuse_deep < len(blocks) - 2 else len(blocks) - 2
            for _ in range(counts):
                need_fuse_blocks.append(blocks.pop(0))
            fused_block = do_block_fusion(need_fuse_blocks)
            logging.debug(
                "fused block, start_name: {}, end_name: {}".format(fused_block.start_name, fused_block.end_name))
            fused_blocks.append(fused_block)
        logging.info("before fuse block size:{}, fused blocks size:{}".format(len(org_blocks), len(fused_blocks)))
        return fused_blocks


class ProgressiveMutation(object):
    """Do progressive mutation."""

    def __init__(self, search_space, model):
        self.block_type_iter = SpaceIter(search_space, 'block_type')
        self.model = model

    def run(self):
        """Run."""
        blocks = match_blocks(self.model)
        target_desc = OrderedDict(copy.deepcopy(self.model.to_desc()))
        fused_block = self.sample_sub_blocks(blocks)
        return mutate_sub_blocks(fused_block, target_desc, self.block_type_iter)

    def sample_sub_blocks_idx(self, block_size):
        """Sample a sub blocks."""
        while True:
            start_idx = random.choice(range(block_size - 2))
            fuse_len = random.choice(range(block_size - start_idx))
            if fuse_len > 2:
                fused_block_idx = [start_idx + 1, start_idx + fuse_len]
                return fused_block_idx

    def sample_sub_blocks(self, blocks):
        """Chose one sub block."""
        s_idx = self.sample_sub_blocks_idx(len(blocks))
        sub_blocks = [block for idx, block in enumerate(blocks) if s_idx[0] < idx < s_idx[1]]
        return do_block_fusion(sub_blocks)


def do_block_fusion(sub_blocks):
    """Do fuse Blocks."""
    nodes = OrderedDict()
    for block in sub_blocks:
        nodes.update(block.nodes)
    fused_block = sub_blocks[0]
    fused_block._nodes = nodes
    fused_block._end_name = sub_blocks[-1].end_name
    logging.info("chose block, start_name: {}, end_name: {}".format(fused_block.start_name, fused_block.end_name))
    return fused_block
