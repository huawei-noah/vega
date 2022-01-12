# -*- coding: utf-8 -*-

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
"""Defined Autoloss class."""
import os
import logging
from vega.common import FileOps, TaskOps
from vega.common.class_factory import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from vega.common import General
from .ada_segment import AdaSegment
from .ada_segment_conf import AdaSegConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class Autoloss(HPOBase):
    """Adjust the weight of mutiple loss dynamically."""

    config = AdaSegConfig()

    def __init__(self, search_space):
        super(Autoloss, self).__init__()
        self.auto_loss = AdaSegment(model_num=self.config.policy.config_count,
                                    total_rungs=self.config.policy.total_rungs,
                                    loss_num=self.config.policy.loss_num)
        self.hpo = self.auto_loss

    def search(self):
        """Search a config."""
        sample = self.auto_loss.propose()
        logging.info("The proposed sample is {}.".format(sample))
        sample_id = sample.get('config_id')
        dynamic_weight = sample.get("dynamic_weight")

        rung_id = sample.get("rung_id")
        return dict(worker_id=sample_id, encoded_desc={"trainer.loss_weight": dynamic_weight}, rung_id=rung_id)

    def update(self, record):
        """Update current performance into hpo score board.

        :param hps: hyper parameters need to update
        :type hps: dict
        """
        rewards = record.get("rewards")
        config_id = record.get('worker_id')
        rung_id = record.get('rung_id')

        worker_path = TaskOps().get_local_worker_path(step_name=record.get("step_name"),
                                                      worker_id=record.get("worker_id"))
        saved_loss = os.path.join(worker_path, "muti_loss.pkl")
        cur_loss = FileOps.load_pickle(saved_loss)

        if not rewards:
            rewards = -1
            logging.error("Update failed because empty performance is got!")

        self.auto_loss.add_score(config_id, rung_id, rewards, cur_loss)
