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
import logging
import copy
import shutil
from vega.algorithms.data_augmentation.common import PBA
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.core.pipeline.hpo_generator import HpoGenerator


@ClassFactory.register(ClassType.HPO)
class PBAHpo(HpoGenerator):
    """An Hpo of PBA, inherit from HpoGenerator."""

    def __init__(self):
        """Init PBAHpo."""
        super(PBAHpo, self).__init__()
        self.transformers = self.cfg.transformers
        self.operation_names = []
        for operation, w_o in self.transformers.items():
            if w_o:
                self.operation_names.append(operation)
        num_operation = len(self.operation_names)
        self.hpo = PBA(self.policy.config_count, self.policy.each_epochs,
                       self.policy.total_rungs, self.local_base_path, num_operation)

    def sample(self):
        """Sample an id and hps from hpo.

        :return: id, hps
        :rtype: int, dict
        """
        re_hps = {}
        sample = self.hpo.propose()
        if sample is not None:
            sample = copy.deepcopy(sample)
            sample_id = sample.get('config_id')
            self._hps_cache[str(sample_id)] = [copy.deepcopy(sample), []]
            trans_para = sample.get('configs')
            re_hps['dataset.transforms'] = [{'type': 'PBATransformer', 'para_array': trans_para,
                                             'operation_names': self.operation_names}]
            checkpoint_path = FileOps.join_path(self.local_base_path, 'cache', 'pba', str(sample_id), 'checkpoint')
            FileOps.make_dir(checkpoint_path)
            if os.path.exists(checkpoint_path):
                re_hps['trainer.checkpoint_path'] = checkpoint_path
            if 'epoch' in sample:
                re_hps['trainer.epochs'] = sample.get('epoch')
            return sample_id, re_hps
        else:
            return None, None

    def update_performance(self, hps, performance):
        """Update current performance into hpo score board.

        :param hps: hyper parameters need to update
        :param performance:  trainer performance
        """
        if isinstance(performance, list) and len(performance) > 0:
            self.hpo.add_score(int(hps.get('config_id')),
                               int(hps.get('rung_id')), performance[0])
        else:
            self.hpo.add_score(int(hps.get('config_id')),
                               int(hps.get('rung_id')), -1)
            logging.error("hpo get empty performance!")
        worker_result_path = self.get_local_worker_path(self.cfg.step_name, str(hps.get('config_id')))
        new_worker_result_path = FileOps.join_path(self.local_base_path, 'cache', 'pba',
                                                   str(hps.get('config_id')), 'checkpoint')
        FileOps.make_dir(worker_result_path)
        FileOps.make_dir(new_worker_result_path)
        if os.path.exists(new_worker_result_path):
            shutil.rmtree(new_worker_result_path)
        shutil.copytree(worker_result_path, new_worker_result_path)

    @property
    def is_completed(self):
        """Make hpo pipe step status is completed.

        :return: hpo status
        :rtype: bool
        """
        return self.hpo.is_completed

    @property
    def best_hps(self):
        """Get best hps."""
        return self.hpo.best_config()

    def _save_score_board(self):
        """Save the internal score board for detail analysis."""
        try:
            self.hpo.sieve_board.to_csv(self._board_file, index=None, header=True)
        except Exception as e:
            logging.error("Failed to save score board file, error={}".format(str(e)))
