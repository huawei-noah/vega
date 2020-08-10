# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""AutoLane algorithm."""
import os
import random
import logging
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.search_space.search_algs.random_search import RandomSearchAlgorithm
from vega.core.common.file_ops import FileOps
from .utils.resnet_variant_det_codec import ResNetVariantDetCodec
from .utils.resnext_variant_det_codec import ResNeXtVariantDetCodec
import pandas as pd
from .conf import AutoLaneConfig
from vega.core.report import Report
from vega.core.common.class_factory import ClassType, ClassFactory


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
        self.random_search = RandomSearchAlgorithm(self.search_space)

    def get_pareto_front(self):
        """Get the pareto front of trained candidates."""
        records = Report().get_pareto_front_records()
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
                sample_desc = self.random_search.search()
                sample['code'] = self.codec.encode(sample_desc)
                sample['method'] = 'random'
            else:
                code_to_mutate = self.get_pareto_front()
                sample['code'] = self.ea_sample(code_to_mutate)
                sample['method'] = 'mutate'
        self.sample_count += 1

        logging.info(sample)
        sample_desc = self.codec.decode(sample)
        sample_desc['detector']['limits'] = {'GFlops': self.config.flops_ceiling_set_by_GFlops}

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
        sample_desc = self.random_search.search()
        sample = self.codec.encode(sample_desc, is_random=True)
        return sample
