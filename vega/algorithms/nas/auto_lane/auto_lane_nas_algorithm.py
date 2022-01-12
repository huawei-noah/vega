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

"""AutoLane algorithm."""
import os
import random
import logging
import pandas as pd
from vega.core.search_algs import SearchAlgorithm
from vega.common import FileOps
from vega.report import ReportServer
from vega.common import ClassType, ClassFactory
from vega.common.config import Config
from vega.common import update_dict
from .utils.resnet_variant_det_codec import ResNetVariantDetCodec
from .utils.resnext_variant_det_codec import ResNeXtVariantDetCodec
from .conf import AutoLaneConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class AutoLaneNas(SearchAlgorithm):
    """AutoLaneNas an algorithm to Search the backbone and ffm to find the best pair.

    :param search_space: input search_space
    :type: SeachSpace
    """

    config = AutoLaneConfig()

    def __init__(self, search_space=None):
        """Init BackboneNas."""
        super(AutoLaneNas, self).__init__(search_space)
        # ea or random
        self.search_space = search_space
        self.num_mutate = self.config.num_mutate
        self.random_ratio = self.config.random_ratio
        self.max_sample = self.config.max_sample
        self.min_sample = self.config.min_sample
        self.sample_count = 0

    def get_pareto_front(self):
        """Get the pareto front of trained candidates."""
        records = ReportServer().get_pareto_front_records()
        codes = []
        for record in records:
            codes.append(record.desc['code'])
        code_to_mutate = random.choice(codes)
        return code_to_mutate

    def get_pareto_list_size(self):
        """Get the number of pareto list."""
        pareto_list_size = 0
        pareto_file_locate = FileOps.join_path(self.local_base_path, "result", "pareto_front.csv")
        if os.path.exists(pareto_file_locate):
            pareto_front_df = pd.read_csv(pareto_file_locate)
            pareto_list_size = pareto_front_df.size
        return pareto_list_size

    @property
    def is_completed(self):
        """Check if NAS is finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search in search_space and return a sample."""
        sample = {}
        while sample is None or 'code' not in sample:
            if random.random() < self.random_ratio or self.get_pareto_list_size() == 0:
                sample_desc = self.search_space.sample()
                sample_desc = self.decode(sample_desc)
                sample['code'] = self.codec.encode(sample_desc['network'])
                sample['method'] = 'random'
            else:
                code_to_mutate = self.get_pareto_front()
                sample['code'] = self.ea_sample(code_to_mutate)
                sample['method'] = 'mutate'
        self.sample_count += 1

        logging.info(sample)
        sample_desc = self.codec.decode(sample)

        return self.sample_count, sample_desc

    def ea_sample(self, code):
        """Run EA algorithm to generate new architecture."""
        backbone_code, ffm_code = code.split('+')
        decoder_map = dict(x=ResNeXtVariantDetCodec, r=ResNetVariantDetCodec)
        CodecSpec = decoder_map.get(backbone_code[0], None)
        if CodecSpec is None:
            raise NotImplementedError(f'Only {decoder_map} is support in auto_lane algorithm')
        backbone_code = CodecSpec(**CodecSpec.EA_sample(backbone_code)).arch_code
        return f'{backbone_code}+{ffm_code}'

    def random_sample(self):
        """Random sample from search_space."""
        sample_desc = self.search_space.sample()
        sample = self.codec.encode(sample_desc, is_random=True)
        return sample

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample

    def decode(self, desc):
        """Decode hps: `trainer.optim.lr : 0.1` to dict format.

        And convert to `vega.common.config import Config` object
        This Config will be override in Trainer or Datasets class
        The override priority is: input hps > user configuration >  default configuration
        :param hps: hyper params
        :return: dict
        """
        hps_dict = {}
        if desc is None:
            return None
        if isinstance(desc, tuple):
            return desc
        for hp_name, value in desc.items():
            hp_dict = {}
            for key in list(reversed(hp_name.split('.'))):
                if hp_dict:
                    hp_dict = {key: hp_dict}
                else:
                    hp_dict = {key: value}
            # update cfg with hps
            hps_dict = update_dict(hps_dict, hp_dict, [])
        return Config(hps_dict)
