# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined AshaHpo class."""
import logging
import copy
from vega.core.search_algs import SearchAlgorithm


class HPOBase(SearchAlgorithm):
    """Base Class for HPO."""

    def __init__(self, search_space=None, **kwargs):
        super(HPOBase, self).__init__(search_space, **kwargs)
        self.hpo = None
        self.search_space = search_space

    @property
    def is_completed(self):
        """Make hpo pipe step status is completed.

        :return: hpo status
        :rtype: bool

        """
        return self.hpo.is_completed

    def search(self):
        """Search an id and hps from hpo.

        :return: id, hps
        :rtype: int, dict
        """
        sample = self.hpo.propose()
        if sample is None:
            return None
        sample = copy.deepcopy(sample)
        sample_id = sample.get('config_id')
        rung_id = sample.get('rung_id')
        desc = sample.get('configs')
        if 'epoch' in sample:
            desc['trainer.epochs'] = sample.get('epoch')
        return dict(worker_id=sample_id, desc=desc, info=rung_id)

    def update(self, record):
        """Update current performance into hpo score board.

        :param record: record need to update.

        """
        rewards = record.get("rewards")
        config_id = record.get('worker_id')
        rung_id = record.get('info')
        if not rewards:
            rewards = -1
            logging.error("hpo get empty performance!")
        if rung_id is not None:
            self.hpo.add_score(config_id, rung_id, rewards)
        else:
            self.hpo.add_score(config_id, rewards)
