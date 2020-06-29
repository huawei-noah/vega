# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Evaluate used to do evaluate process."""
import os
import copy
import logging
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.distributed_worker import DistributedWorker
from vega.core.trainer.utils import WorkerTypes


@ClassFactory.register(ClassType.EVALUATOR)
class Evaluator(DistributedWorker):
    """Evaluator.

    :param worker_info: worker_info
    :type worker_info: dict, default to None
    """

    def __init__(self, worker_info=None):
        """Init Evaluator."""
        super(Evaluator, self).__init__(self.cfg)
        Evaluator.__worker_id__ = Evaluator.__worker_id__ + 1
        self._worker_id = Evaluator.__worker_id__
        # for init ids
        self.worker_type = WorkerTypes.EVALUATOR
        self.worker_info = worker_info
        if worker_info is not None:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        # main evalutors setting
        self.sub_worker_list = []

    @property
    def size(self):
        """Return the size of current evaluator list."""
        return len(self.sub_worker_list)

    def add_evaluator(self, evaluator):
        """Add a sub-evaluator to this evaluator.

        :param evaluator: Description of parameter `evaluator`.
        :type evaluator: object,
        """
        if not isinstance(evaluator, DistributedWorker):
            return
        elif evaluator.worker_type is not None:
            sub_evaluator = copy.deepcopy(evaluator)
            sub_evaluator.worker_info = self.worker_info
            if self.worker_info is not None:
                sub_evaluator.step_name = self.worker_info["step_name"]
                sub_evaluator.worker_id = self.worker_info["worker_id"]
            self.sub_worker_list.append(sub_evaluator)
        return

    def set_worker_info(self, worker_info):
        """Set current evaluator's worker_info.

        :param worker_info: Description of parameter `worker_info`.
        :type worker_info: dict,
        """
        if worker_info is None:
            raise ValueError("worker_info should not be None type!")
        self.worker_info = worker_info
        self.step_name = self.worker_info["step_name"]
        self.worker_id = self.worker_info["worker_id"]

        for sub_evaluator in self.sub_worker_list:
            sub_evaluator.worker_info = self.worker_info
            sub_evaluator.step_name = self.worker_info["step_name"]
            sub_evaluator.worker_id = self.worker_info["worker_id"]
        return
