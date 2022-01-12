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

"""Mutate search algorithm used to search BiSeNet code."""
import random
import copy
import logging
import numpy as np
from vega.report import ReportServer
from .conf import SegmentationConfig


class SegmentationMutate(object):
    """Mutate algorithm of SegmentationNas."""

    config = SegmentationConfig()

    def __init__(self, search_space=None):
        """Construct the SegmentationMutate class.

        :param search_space: Config of the search space.
        """
        self.num_transform = self.config.num_transform

    def _insert(self, arc):
        """Insert method of mutate algorithm.

        :param arc: Code of the BiSeNet to mutate.
        :return: inserted code.
        """
        idx = np.random.randint(low=1, high=len(arc))
        arc = arc[:idx] + '1' + arc[idx:]
        print('insert', idx)
        return arc

    def _remove(self, arc):
        """Remove method of mutate algorithm.

        :param arc: Code of the BiSeNet to mutate.
        :return: removed code.
        """
        is_not_valid = True
        arch_original = copy.copy(arc)
        count = 0
        one_index = [i for i, a in enumerate(arc) if a == '1']
        while is_not_valid:
            arc = arch_original
            idx = np.random.choice(one_index)
            arc = arc[0:idx] + arc[idx + 1:]
            is_not_valid = (not self.is_valid(arc))
            count += 1
            if count > 100:
                return arch_original
        print('remove', idx)
        return arc

    def _swap(self, arc):
        """Swap method of mutate algorithm.

        :param arc: Code of the BiSeNet to mutate.
        :return: swapped code.
        """
        is_not_valid = True
        arc_origin = copy.copy(arc)
        count = 0
        while is_not_valid or arc == arc_origin:
            idx = np.random.randint(low=0, high=len(arc) - 1)
            arc = arc[:idx] + arc[idx + 1] + arc[idx] + arc[idx + 2:]
            is_not_valid = (not self.is_valid(arc))
            count += 1
            if count > 100:
                return arc_origin
        print('swap', idx)
        return arc

    def _remove2(self, arc):
        """Remove method of mutate algorithm.

        :param arc: Code of the BiSeNet to mutate.
        :return: removed code.
        """
        is_not_valid = True
        arch_original = copy.copy(arc)
        count = 0
        two_index = [i for i, a in enumerate(arc) if a == '2']
        while is_not_valid:
            arc = arch_original
            idx = np.random.choice(two_index)
            arc = arc[0:idx] + arc[idx + 1:]
            is_not_valid = (not self.is_valid(arc))
            count += 1
            if count > 100:
                return arch_original
        print('remove', idx)
        return arc

    def _append2(self, arc):
        """Append method of mutate algorithm.

        :param arc: Code of the BiSeNet to mutate.
        :return: appended code.
        """
        return arc + '2'

    def is_valid(self, arc):
        """Judge whether the code to mutate is valid.

        :param arc: Code of the BiSeNet to mutate.
        :return: True means the code is valid, otherwise False.
        """
        stages = arc.split('-')
        for stage in stages:
            if len(stage) == 0:
                return False
        return True

    def mutate_channel(self, spatial_path):
        """Mutate channel of spatial path.

        :param spatial_path: Code of the BiSeNet's spatial path.
        :return: transformed spatial path.
        """
        if np.random.random() < 0.2:
            if int(spatial_path[0]) == 32:
                spatial_path[0] = '64'
                spatial_path[-1] = self._remove2(spatial_path[-1])
                if spatial_path[-1].count('2') > 1:
                    spatial_path[0] = '32'
            else:
                spatial_path[0] = '32'
                spatial_path[-1] = self._append2(spatial_path[-1])
        return spatial_path

    def do_transform(self, arc, num_mutate):
        """Do transforms to the input code.

        :param arc: Code to transform.
        :return: transformed code.
        """
        for i in range(num_mutate):
            op_idx = np.random.randint(low=0, high=3)
            if op_idx == 0:
                arc = self._insert(arc)
            elif op_idx == 1:
                arc = self._remove(arc)
            elif op_idx == 2:
                arc = self._swap(arc)
            else:
                raise Exception('operation index out of range')
        return arc

    def search(self):
        """Search code of one model.

        :return: searched code of the model.
        """
        records = ReportServer().get_pareto_front_records(['nas'])
        encodings = []
        for record in records:
            custom = record.desc['custom']
            encodings.append(custom['encoding'])
        pareto_front = encodings
        model_str = random.choice(pareto_front)
        print('model_str', model_str)
        ratio = np.random.randint(low=0, high=self.num_transform + 1)
        print('ratio', ratio)
        context_path = model_str[0].split('_')
        spatial_path = model_str[1].split('_')
        print('context_path', context_path)
        print('spatial_path', spatial_path)
        spatial_path = self.mutate_channel(spatial_path)
        context_path[-1] = self.do_transform(context_path[-1], num_mutate=ratio)
        spatial_path[-1] = self.do_transform(spatial_path[-1], num_mutate=self.num_transform - ratio)
        encoding = ['_'.join(context_path), '_'.join(spatial_path)]
        logging.info("Mutate from {} to {}".format(model_str, encoding))
        return encoding
