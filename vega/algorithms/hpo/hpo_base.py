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

"""Defined AshaHpo class."""

import logging
import copy
from threading import Lock
from vega.core.search_algs import SearchAlgorithm
from vega.report.record import ReportRecord
from vega.common.message_server import MessageServer


__all__ = ["HPOBase"]
_instance = None
_lock = Lock()


class HPOBase(SearchAlgorithm):
    """Base Class for HPO."""

    def __init__(self, search_space=None, **kwargs):
        super(HPOBase, self).__init__(search_space, **kwargs)
        self.hpo = None
        self.search_space = search_space
        global _instance
        _instance = self
        MessageServer().register_handler("next_rung", next_rung)

    @property
    def is_completed(self):
        """Make hpo pipe step status is completed.

        :return: hpo status
        :rtype: bool

        """
        return self.hpo.is_completed

    def search(self, config_id=None):
        """Search an id and hps from hpo.

        :return: id, hps
        :rtype: int, dict
        """
        global _lock
        with _lock:
            if config_id is not None:
                sample = self.hpo.next_rung(config_id)
            else:
                sample = self.hpo.propose()
            if sample is None:
                return None
            sample = copy.deepcopy(sample)
            sample_id = sample.get('config_id')
            rung_id = sample.get('rung_id')
            desc = sample.get('configs')
            if 'epoch' in sample:
                desc['trainer.epochs'] = sample.get('epoch')
            return dict(worker_id=sample_id, encoded_desc=desc, rung_id=rung_id)

    def update(self, record):
        """Update current performance into hpo score board.

        :param record: record need to update.

        """
        global _lock
        with _lock:
            rewards = record.get("rewards")
            config_id = int(record.get('worker_id'))
            rung_id = record.get('rung_id')
            if rewards is None:
                rewards = -1 * float('inf')
                logging.error("hpo get empty performance!")
            if rung_id is not None:
                self.hpo.add_score(config_id, int(rung_id), rewards)
            else:
                self.hpo.add_score(config_id, rewards)


def next_rung(**kwargs):
    """Prompt next rung."""
    global _instance
    if _instance is None or kwargs is None:
        return {"result": "success", "data": {"rung_id": None, "message": "instance is none"}}
    if not hasattr(_instance.hpo, "next_rung"):
        return {"result": "success", "data": {"rung_id": None, "message": "do not has next_rung method"}}

    record = ReportRecord().load_dict(kwargs)
    if None in record.rewards:
        return {"result": "success", "data": {"rung_id": None, "message": "part of rewards is missing"}}

    _instance.update(record.serialize())
    result = _instance.search(config_id=record.worker_id)
    if result is not None and "encoded_desc" in result and "trainer.epochs" in result["encoded_desc"]:
        data = {"rung_id": result["rung_id"], "epochs": result["encoded_desc"]["trainer.epochs"]}
        return {"result": "success", "data": data}
    else:
        return {"result": "success", "data": {"rung_id": None, "message": result}}
