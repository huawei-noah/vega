# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The search algorithm of SPNAS."""

import os
import random
import numpy as np
import logging
import copy
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.search_space.codec import Codec
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.algorithms.nas.sp_nas.utils import ListDict


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SpNas(SearchAlgorithm):
    """Search Algorithm Stage of SPNAS."""

    def __init__(self, search_space=None):
        super(SpNas, self).__init__(search_space)
        self.search_space = search_space
        self.codec = Codec(self.cfg.codec, search_space)
        self.sample_level = self.cfg.sample_level
        self.max_sample = self.cfg.max_sample
        self.max_optimal = self.cfg.max_optimal
        self._total_list_name = self.cfg.total_list
        self.serial_settings = self.cfg.serial_settings

        self._total_list = ListDict()
        self.sample_count = 0
        self.init_code = None
        remote_output_path = FileOps.join_path(self.local_output_path, self.cfg.step_name)

        if 'last_search_result' in self.cfg:
            last_search_file = self.cfg.last_search_result
            assert FileOps.exists(os.path.join(remote_output_path, last_search_file)
                                  ), "Not found serial results!"
            # self.download_task_folder()
            last_search_results = os.path.join(self.local_output_path, last_search_file)
            last_search_results = ListDict.load_csv(last_search_results)
            pre_worker_id, pre_arch = self.select_from_remote(self.max_optimal, last_search_results)
            # re-write config template
            if self.cfg.regnition:
                self.codec.config_template['model']['backbone']['reignition'] = True
                assert FileOps.exists(os.path.join(remote_output_path,
                                                   pre_arch + '_imagenet.pth')
                                      ), "Not found {} pretrained .pth file!".format(pre_arch)
                pretrained_pth = os.path.join(self.local_output_path, pre_arch + '_imagenet.pth')
                self.codec.config_template['model']['pretrained'] = pretrained_pth
                pre_worker_id = -1
            # update config template
            self.init_code = dict(arch=pre_arch,
                                  pre_arch=pre_arch.split('_')[1],
                                  pre_worker_id=pre_worker_id)

        logging.info("inited SpNas {}-level search...".format(self.sample_level))

    @property
    def is_completed(self):
        """Check sampling if finished.

        :return: True is completed, or False otherwise
        :rtype: bool
        """
        return self.sample_count > self.max_sample

    @property
    def num_samples(self):
        """Get the number of sampled architectures.

        :return: The number of sampled architectures
        :rtype: int
        """
        return len(self._total_list)

    def select_from_remote(self, max_optimal, total_list):
        """Select base model to mutate.

        :return: worker id and arch encode
        :rtype: int, str
        """
        def normalization(x):
            sum_ = sum(x)
            return [float(i) / float(sum_) for i in x]

        # rank with mAP and memory
        top_ = total_list.sort('mAP')[:max_optimal]
        if max_optimal > 1:
            prob = [round(np.log(i + 1e-2), 2) for i in range(1, len(top_) + 1)]
            prob_temp = prob
            sorted_ind = sorted(range(len(top_)), key=lambda k: top_['memory'][k], reverse=True)
            for idx, ind in enumerate(sorted_ind):
                prob[ind] += prob_temp[idx]
            ind = np.random.choice(len(top_), p=normalization(prob))
            worker_id, arch = top_['worker_id', 'arch'][ind]
        else:
            worker_id, arch = top_['worker_id', 'arch'][0]
        return worker_id, arch

    def search(self):
        """Search a sample.

        :return: sample count and info
        :rtype: int, dict
        """
        code = self.init_code
        if self.num_samples > 0:
            pre_worker_id, pre_arch = self.select_from_remote(self.max_optimal, self._total_list)
            block_type, pre_serial, pre_paral = pre_arch.split('_')

            success = False
            while not success:
                serialnet, parallelnet = pre_serial, pre_paral
                if self.sample_level == 'serial':
                    serialnet = self._mutate_serialnet(serialnet, **self.serial_settings)
                    parallelnet = '-'.join(['0'] * len(serialnet.split('-')))
                elif self.sample_level == 'parallel':
                    parallelnet = self._mutate_parallelnet(parallelnet)
                    pre_worker_id = self.init_code['pre_worker_id']
                else:
                    raise ValueError("Unknown type of sample level")
                arch = self.codec.encode(block_type, serialnet, parallelnet)
                if arch not in self._total_list['arch'] or len(self._total_list['arch']) == 0:
                    success = True
            code = dict(arch=arch,
                        pre_arch=pre_serial,
                        pre_worker_id=pre_worker_id)

        self.sample_count += 1
        logging.info("The {}-th successfully sampling result: {}".format(self.sample_count, code))
        net_desc = self.codec.decode(code)
        return self.sample_count, net_desc

    def update(self, worker_result_path):
        """Update sampler."""
        performance_file = self.performance_path(worker_result_path)
        logging.info(
            "SpNas.update(), performance file={}".format(performance_file))
        info = FileOps.load_pickle(performance_file)
        if info is not None:
            self._total_list.append(info)
        else:
            logging.info("SpNas.update(), file is not exited, "
                         "performance file={}".format(performance_file))
        self.save_output(self.local_output_path)
        if self.backup_base_path is not None:
            FileOps.copy_folder(self.local_output_path, self.backup_base_path)

    def performance_path(self, worker_result_path):
        """Get performance path."""
        performance_dir = os.path.join(worker_result_path, 'performance')
        if not os.path.exists(performance_dir):
            FileOps.make_dir(performance_dir)
        return os.path.join(performance_dir, 'performance.pkl')

    def save_output(self, local_output_path):
        """Save results."""
        local_totallist_path = os.path.join(local_output_path, self._total_list_name)
        self._total_list.to_csv(local_totallist_path)

    def _mutate_serialnet(self, arch, num_mutate=3, expend_ratio=0, addstage_ratio=0.95, max_stages=6):
        """Swap & Expend operation in Serial-level searching.

        :param arch: base arch encode
        :type arch: str
        :param num_mutate: number of mutate
        :type num_mutate: int
        :param expend_ratio: probability of expend block
        :type expend_ratio: float
        :param addstage_ratio: probability of expand new stage
        :type addstage_ratio: float
        :param max_stages:  max stage allowed to expand
        :type max_stages: int
        :return: arch encode after mutate
        :rtype: str
        """
        def is_valid(arch):
            stages = arch.split('-')
            for stage in stages:
                if len(stage) == 0:
                    return False
            return True

        def expend(arc):
            idx = np.random.randint(low=1, high=len(arc))
            arc = arc[:idx] + '1' + arc[idx:]
            return arc, idx

        def swap(arc, len_step=3):
            is_not_valid = True
            arc_origin = copy.deepcopy(arc)
            temp = arc.split('-')
            num_insert = len(temp) - 1
            while is_not_valid or arc == arc_origin:
                next_start = 0
                arc = list(''.join(temp))
                for i in range(num_insert):
                    pos = arc_origin[next_start:].find('-') + next_start
                    assert arc_origin[pos] == '-', "Wrong '-' is found!"
                    max_step = min(len_step, max(len(temp[i]), len(temp[i + 1])))
                    step_range = list(range(-1 * max_step, max_step))
                    step = random.choice(step_range)
                    next_start = pos + 1
                    pos = pos + step
                    arc.insert(pos, '-')
                arc = ''.join(arc)
                is_not_valid = (not is_valid(arc))
            return arc

        arch_origin = arch
        success = False
        k = 0
        while not success:
            k += 1
            arch = arch_origin
            ops = []
            for i in range(num_mutate):
                op_idx = np.random.randint(low=0, high=3)
                adds_thresh_ = addstage_ratio if len(arch.split('-')) < max_stages else 1
                if op_idx == 0 and random.random() > expend_ratio:
                    arch, idx = expend(arch)
                    arch, idx = expend(arch)
                elif op_idx == 1:
                    arch = swap(arch)
                elif op_idx == 2 and random.random() > adds_thresh_:
                    arch = arch + '-1'
                    ops.append('add stage')
                else:
                    ops.append('Do Nothing.')
            success = arch != arch_origin
            flag = 'Success' if success else 'Failed'
            logging.info('Serial-level Sample{}: {}. {}.'.format(k + 1, ' -> '.join(ops), flag))
        return arch

    def _mutate_parallelnet(self, arch):
        """Mutate operation in Parallel-level searching.

        :param arch: base arch encode
        :type arch: str
        :return: parallel arch encode after mutate
        :rtype: str
        """
        def limited_random(num_stage):
            p = [0.4, 0.3, 0.2, 0.1]
            l = np.random.choice(4, size=num_stage, replace=True, p=p)
            l = [str(i) for i in l]
            return '-'.join(l)

        num_stage = len(arch.split('-'))
        success = False
        k = 0
        while not success:
            k += 1
            new_arch = limited_random(num_stage)
            success = new_arch != arch
            flag = 'Success' if success else 'Failed'
            logging.info('Parallel-level Sample{}: {}.'.format(k + 1, flag))
        return new_arch
