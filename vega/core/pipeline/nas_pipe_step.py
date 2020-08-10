# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Nas Pipe Step defined in Pipeline."""
import logging
import time
import traceback
from .pipe_step import PipeStep
from .generator import Generator
from ..scheduler.master import Master
from ..common.class_factory import ClassFactory, ClassType
from vega.core.common.loader import load_conf_from_desc
from vega.search_space.networks.network_desc import NetworkDesc
from ..pipeline.conf import PipeStepConfig
from vega.core.report import Report
from vega.core.common.general import General


@ClassFactory.register(ClassType.PIPE_STEP)
class NasPipeStep(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.generator = Generator()
        self.master = Master()
        self.need_evaluate = self._is_existed_evaluator()

    def do(self):
        """Do the main task in this pipe step."""
        logging.debug("NasPipeStep started...")
        while not self.generator.is_completed:
            res = self.generator.sample()
            if res:
                self._dispatch_trainer(res)
            else:
                time.sleep(0.5)
            self._after_train(wait_until_finish=False)
        self.master.join()
        self._after_train(wait_until_finish=True)
        logging.debug("Pareto_front values: %s", Report().pareto_front(General.step_name))
        Report().output_pareto_front(General.step_name)
        self.master.close_client()

    def _after_train(self, wait_until_finish):
        all_finished_trainer_info = self.master.pop_all_finished_train_worker()
        if self.need_evaluate:
            if all_finished_trainer_info:
                self._dispatch_evaluator(all_finished_trainer_info)
            if wait_until_finish:
                self.master.join()
            all_finished_evaluator_info = self.master.pop_all_finished_evaluate_worker()
            if all_finished_evaluator_info:
                self.update_generator(self.generator, all_finished_evaluator_info)
        else:
            self.update_generator(self.generator, all_finished_trainer_info)

    def _dispatch_trainer(self, samples):
        for (id_ele, desc) in samples:
            cls_trainer = ClassFactory.get_cls('trainer')
            load_conf_from_desc(PipeStepConfig, desc)
            model_ele = NetworkDesc(desc).to_model()
            trainer = cls_trainer(model_ele, id_ele, hps=desc)
            logging.info("submit trainer, id={}".format(id_ele))
            self.master.run(trainer)
        if isinstance(samples, list) and len(samples) > 1:
            self.master.join()

    def _dispatch_evaluator(self, finished_traineres):
        try:
            cls_evaluator = ClassFactory.get_cls('evaluator')
        except Exception as e:
            cls_evaluator = None
            logging.warning("Get evaluator failed:{}".format(str(e)))
        if cls_evaluator is not None:
            for work_info in finished_traineres:
                evaluator = cls_evaluator(work_info)
                logging.info("submit evaluator, step_name={}, worker_id={}".format(
                    work_info.get("step_name"), work_info.get("worker_id")))
                self.master.run(evaluator)

    def _is_existed_evaluator(self):
        try:
            ClassFactory.get_cls('evaluator')
            return True
        except Exception:
            return False

    def update_generator(self, generator, worker_info):
        """Get finished worker's info, and use it to update target `generator`.

        Will get the finished worker's working dir, and then call the function
        `generator.update(step_name, worker_id)`.
        :param Generator generator: The target `generator` need to update.
        :param worker_info: `worker_info` is the finished worker's info, usually
            a dict or list of dict include `step_name` and `worker_id`.
        :type worker_info: dict or list of dict.

        """
        if worker_info is None:
            return
        if not isinstance(worker_info, list):
            worker_info = [worker_info]
        for one_info in worker_info:
            step_name = one_info["step_name"]
            worker_id = one_info["worker_id"]
            logging.info("update generator, step name: {}, worker id: {}".format(step_name, worker_id))
            try:
                generator.update(step_name, worker_id)
            except Exception:
                logging.error("Failed to upgrade generator, step_name={}, worker_id={}.".format(step_name, worker_id))
                logging.error(traceback.format_exc())
