# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined a basic HpoGenerator class."""
import json
import os
import csv
import logging
import copy
import traceback
from vega.core.common.task_ops import TaskOps
from vega.core.common.file_ops import FileOps
from vega.core.common.utils import update_dict


logger = logging.getLogger(__name__)


class HpoGenerator(TaskOps):
    """HpoGenerator, a base class for different hpo class."""

    __hpo_id__ = 0

    def __init__(self):
        """Init HpoBase."""
        super(HpoGenerator, self).__init__(self.cfg)
        self.hpo = None
        self.policy = self.cfg.get('policy')
        self._hps_cache = {}
        step_path = FileOps.join_path(self.local_output_path, self.cfg.step_name)
        self._best_hps_file = FileOps.join_path(step_path, 'best_hps.json')
        self._cache_file = FileOps.join_path(step_path, 'cache.csv')
        self._board_file = FileOps.join_path(step_path, 'score_board.csv')

    def proposal(self):
        """Proposal interface function to proposal one hps.

        :return: id and hps
        :rtype: str and dict

        """
        raise NotImplementedError

    @property
    def is_completed(self):
        """Make hpo pipe step status is completed.

        :return: hpo status
        :rtype: bool

        """
        raise NotImplementedError

    @property
    def best_hps(self):
        """Get best hps."""
        raise NotImplementedError

    def update_performance(self, hps, performance):
        """Update performance value into hpo algorithms.

        :param hps: hps cached in `self._hps_cache`
        :param performance: performance value from task file `performance.txt`

        """
        raise NotImplementedError

    def sample(self):
        """Call proposal method to proposal one hps.

        hp0 id will be provided to the hps and save in hps_cache

        :return: id and hps
        :rtype: str and dict

        """
        NotImplementedError

    def update(self, step_name, worker_id):
        """Update hpo score into score board.

        :param step_name: step name in pipeline
        :param worker_id: worker id of worker

        """
        worker_id = str(worker_id)
        performance = self._get_performance(step_name, worker_id)
        if worker_id in self._hps_cache:
            hps = self._hps_cache[worker_id][0]
            self._hps_cache[worker_id][1] = copy.deepcopy(performance)
            logging.info("get hps need to update, worker_id=%s, hps=%s", worker_id, str(hps))
            self.update_performance(hps, performance)
            logging.info("hpo_id=%s, hps=%s, performance=%s", worker_id, str(hps), str(performance))
            self._save_hpo_cache()
            self._save_score_board()
            self._save_best()
            if self.need_backup and self.backup_base_path is not None:
                FileOps.copy_folder(self.local_output_path,
                                    FileOps.join_path(self.backup_base_path, self.output_subpath))
            logger.info("Hpo update finished.")
        else:
            logger.error("worker_id not in hps_cache.")

    def _decode_best_hps(self):
        """Decode best hps: `trainer.optim.lr : 0.1` to dict format.

        :return: dict
        """
        hps = self.best_hps['configs']
        hps_dict = {}
        for hp_name, value in hps.items():
            hp_dict = {}
            for key in list(reversed(hp_name.split('.'))):
                if hp_dict:
                    hp_dict = {key: hp_dict}
                else:
                    hp_dict = {key: value}
            # update cfg with hps
            hps_dict = update_dict(hps_dict, hp_dict)
        return hps_dict

    def _save_best(self):
        """Save current best config or hyper params."""
        try:
            with open(self._best_hps_file, 'w') as f:
                f.write(json.dumps(self._decode_best_hps()))
        except Exception:
            logger.error("Failed to save best hps file, file={}".format(self._best_hps_file))
            logging.error(traceback.format_exc())

    def _save_hpo_cache(self):
        """Save all hpo info."""
        csv_columns = ['id', 'hps', 'performance']

        try:
            FileOps.make_base_dir(self._cache_file)
            with open(self._cache_file, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                writer.writeheader()
                for hpo_id, value in self._hps_cache.items():
                    data = {'id': hpo_id, 'hps': value[0], 'performance': value[1]}
                    writer.writerow(data)
        except Exception:
            logger.error("Failed to save hpo cache, file={}".format(self._cache_file))
            logging.error(traceback.format_exc())

    def _save_score_board(self):
        """Save the internal score board for detail analysis."""
        pass

    def _get_performance(self, step_name, worker_id):
        """Read Performance values from perform.txt.

        :param step_name: step name in the pipeline.
        :type step_name: str.
        :param worker_id: the worker's worker id.
        :type worker_id: str.
        :return: performance value
        :rtype: int/float/list

        """
        _file = FileOps.join_path(self.get_local_worker_path(step_name, worker_id), "performance.txt")
        if not os.path.isfile(_file):
            logger.info("Performance file is not exited, file={}".format(_file))
            return []
        with open(_file, 'r') as f:
            performance = []
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                data = json.loads(line)
                if isinstance(data, list):
                    data = data[0]
                performance.append(data)
            logger.info("performance={}".format(performance))
        return performance
