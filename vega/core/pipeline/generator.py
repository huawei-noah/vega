# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Generator for NasPipeStep."""
import logging
import os
import pickle
from copy import deepcopy
from vega.core.search_algs import SearchAlgorithm
from vega.core.search_space.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig
from zeus.common.general import General
from zeus.common.task_ops import TaskOps
from zeus.report import Report, ReportRecord
from zeus.common.config import Config
from zeus.common import update_dict
from zeus.quota.quota_compare import QuotaCompare


class Generator(object):
    """Convert search space and search algorithm, sample a new model."""

    def __init__(self):
        self.step_name = General.step_name
        self.search_space = SearchSpace()
        self.search_alg = SearchAlgorithm(self.search_space)
        self.report = Report()
        self.record = ReportRecord()
        self.record.step_name = self.step_name
        if hasattr(self.search_alg.config, 'objective_keys'):
            self.record.objective_keys = self.search_alg.config.objective_keys
        self.quota = QuotaCompare('restrict')

    @property
    def is_completed(self):
        """Define a property to determine search algorithm is completed."""
        return self.search_alg.is_completed or self.quota.is_halted()

    def sample(self):
        """Sample a work id and model from search algorithm."""
        res = self.search_alg.search()
        if not res:
            return None
        if not isinstance(res, list):
            res = [res]
        if len(res) == 0:
            return None
        out = []
        for sample in res:
            desc = sample.get("desc") if isinstance(sample, dict) else sample[1]
            desc = self._decode_hps(desc)
            model_desc = deepcopy(desc)
            if "modules" in desc:
                PipeStepConfig.model.model_desc = deepcopy(desc)
            elif "network" in desc:
                origin_desc = PipeStepConfig.model.model_desc
                desc = update_dict(desc["network"], origin_desc)
                PipeStepConfig.model.model_desc = deepcopy(desc)
            if self.quota.is_filtered(desc):
                continue
            record = self.record.from_sample(sample, desc)
            Report().broadcast(record)
            out.append((record.worker_id, model_desc))
        return out

    def update(self, step_name, worker_id):
        """Update search algorithm accord to the worker path.

        :param step_name: step name
        :param worker_id: current worker id
        :return:
        """
        report = Report()
        record = report.receive(step_name, worker_id)
        logging.debug("Get Record=%s", str(record))
        self.search_alg.update(record.serialize())
        report.dump_report(record.step_name, record)
        self.dump()
        logging.info("Update Success. step_name=%s, worker_id=%s", step_name, worker_id)
        logging.info("Best values: %s", Report().print_best(step_name=General.step_name))

    @staticmethod
    def _decode_hps(hps):
        """Decode hps: `trainer.optim.lr : 0.1` to dict format.

        And convert to `zeus.common.config import Config` object
        This Config will be override in Trainer or Datasets class
        The override priority is: input hps > user configuration >  default configuration
        :param hps: hyper params
        :return: dict
        """
        hps_dict = {}
        if hps is None:
            return None
        if isinstance(hps, tuple):
            return hps
        for hp_name, value in hps.items():
            hp_dict = {}
            for key in list(reversed(hp_name.split('.'))):
                if hp_dict:
                    hp_dict = {key: hp_dict}
                else:
                    hp_dict = {key: value}
            # update cfg with hps
            hps_dict = update_dict(hps_dict, hp_dict, [])
        return Config(hps_dict)

    def dump(self):
        """Dump generator to file."""
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".generator")
        with open(_file, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def restore(cls):
        """Restore generator from file."""
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".generator")
        if os.path.exists(_file):
            with open(_file, "rb") as f:
                return pickle.load(f)
        else:
            return None
