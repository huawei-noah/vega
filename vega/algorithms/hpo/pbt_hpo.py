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

"""Defined PBTHpo class."""
import os
import copy
import shutil
import logging
import numpy as np
from vega.algorithms.hpo.sha_base.pbt import PBT
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.algorithms.hpo.hpo_base import HPOBase
from .pbt_conf import PBTConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class PBTHpo(HPOBase):
    """An Hpo of PBT."""

    config = PBTConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init PBTHpo."""
        self.search_space = search_space
        super(PBTHpo, self).__init__(search_space, **kwargs)
        self.hyperparameter_list = self.get_hyperparameters(self.config.policy.config_count)
        self.hpo = PBT(self.config.policy.config_count, self.config.policy.each_epochs,
                       self.config.policy.total_rungs, self.local_base_path,
                       paras_list=self.hyperparameter_list)

    def get_hyperparameters(self, num):
        """Use the trained model to propose a set of params from SearchSpace.

        :param int num: number of random samples from hyperparameter space.
        :return: list of random sampled config from hyperparameter space.
        :rtype: list.

        """
        params_list = []
        for _ in range(num):
            parameters = self.search_space.get_sample_space()
            if parameters is None:
                return None
            predictions = np.random.rand(parameters.shape[0], 1)
            index = np.argmax(predictions)
            param = self.search_space.decode(parameters[index, :])
            params_list.append(param)
        return params_list

    def search(self):
        """Search an id and hps from hpo."""
        sample = self.hpo.propose()
        if sample is None:
            return None
        re_hps = {}
        sample = copy.deepcopy(sample)
        sample_id = sample.get('config_id')
        cur_configs = sample.get('configs')
        all_configs = sample.get("all_configs")
        rung_id = sample.get('rung_id')

        checkpoint_path = FileOps.join_path(self.local_base_path, 'cache', str(sample_id), 'checkpoint')
        FileOps.make_dir(checkpoint_path)
        if os.path.exists(checkpoint_path):
            re_hps['trainer.checkpoint_path'] = checkpoint_path
        if 'epoch' in sample:
            re_hps['trainer.epochs'] = sample.get('epoch')
        re_hps.update(cur_configs)
        re_hps['trainer.all_configs'] = all_configs
        logging.info("Current rung [ {} /{}] ".format(rung_id, self.config.policy.total_rungs))
        return dict(worker_id=sample_id, encoded_desc=re_hps, rung_id=rung_id)

    def update(self, record):
        """Update current performance into hpo score board.

        :param hps: hyper parameters need to update
        :param performance:  trainer performance
        """
        super().update(record)
        config_id = str(record.get('worker_id'))
        step_name = record.get('step_name')
        worker_result_path = self.get_local_worker_path(step_name, config_id)
        new_worker_result_path = FileOps.join_path(self.local_base_path, 'cache', config_id, 'checkpoint')
        FileOps.make_dir(worker_result_path)
        FileOps.make_dir(new_worker_result_path)
        if os.path.exists(new_worker_result_path):
            shutil.rmtree(new_worker_result_path)
        shutil.copytree(worker_result_path, new_worker_result_path)
