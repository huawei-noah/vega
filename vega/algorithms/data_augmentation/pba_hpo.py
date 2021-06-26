# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined PBAHpo class."""
import os
import copy
import shutil
from vega.algorithms.data_augmentation.common import PBA
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.algorithms.hpo.hpo_base import HPOBase
from .pba_conf import PBAConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class PBAHpo(HPOBase):
    """An Hpo of PBA."""

    config = PBAConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init PBAHpo."""
        super(PBAHpo, self).__init__(search_space, **kwargs)
        self.transformers = search_space.transformers
        self.operation_names = [list(op.keys())[0] for op in self.transformers if list(op.values())[0]]
        num_operation = len(self.operation_names)
        self.hpo = PBA(self.config.policy.config_count, self.config.policy.each_epochs,
                       self.config.policy.total_rungs, self.local_base_path, num_operation)

    def search(self):
        """Search an id and hps from hpo."""
        sample = self.hpo.propose()
        if sample is None:
            return None
        re_hps = {}
        sample = copy.deepcopy(sample)
        sample_id = sample.get('config_id')
        trans_para = sample.get('configs')
        rung_id = sample.get('rung_id')
        all_para = sample.get('all_configs')
        re_hps['dataset.transforms'] = [{'type': 'PBATransformer', 'para_array': trans_para, 'all_para': all_para,
                                         'operation_names': self.operation_names}]
        checkpoint_path = FileOps.join_path(self.local_base_path, 'worker', 'cache', str(sample_id), 'checkpoint')
        FileOps.make_dir(checkpoint_path)
        if os.path.exists(checkpoint_path):
            re_hps['trainer.checkpoint_path'] = checkpoint_path
        if 'epoch' in sample:
            re_hps['trainer.epochs'] = sample.get('epoch')
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
        new_worker_result_path = FileOps.join_path(self.local_base_path, 'worker', 'cache', config_id, 'checkpoint')
        FileOps.make_dir(worker_result_path)
        FileOps.make_dir(new_worker_result_path)
        if os.path.exists(new_worker_result_path):
            shutil.rmtree(new_worker_result_path)
        shutil.copytree(worker_result_path, new_worker_result_path)
