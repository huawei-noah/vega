# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Hpo_pipe_step defined the HpoPipeStep class."""

import logging
import traceback
from ..common.class_factory import ClassFactory, ClassType
from ..scheduler.master import Master
from .pipe_step import PipeStep
from vega.core.common.config import Config
from vega.core.common.utils import update_dict

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class HpoPipeStep(PipeStep):
    """HpoPipeStep for main step of hpo."""

    def __init__(self):
        """Init method for class `HpoPipeStep`."""
        super().__init__()
        self.master = Master()
        self.hpo = ClassFactory.get_cls(ClassType.HPO)()

    def do(self):
        """Do the main method for hpo pipestep."""
        logger.info("HpoPipeStep started...")
        use_evaluator = False
        while not self.hpo.is_completed:
            hpo_id, hps = self.hpo.sample()
            if isinstance(hpo_id, list) and isinstance(hps, list):
                for hpo_id_ele, hps_ele in zip(hpo_id, hps):
                    logger.info("id=%s, hps=%s", hpo_id_ele, hps_ele)
                    cls_trainer = ClassFactory.get_cls(ClassType.TRAINER)
                    hps_ele = HpoPipeStep._decode_hps(hps_ele)
                    trainer = cls_trainer(hps=hps_ele, id=hpo_id_ele, load_checkpoin=True)
                    self.master.run(trainer)
                self.master.join()
            elif hps is not None:
                logger.info("id=%s, hps=%s", hpo_id, hps)
                cls_trainer = ClassFactory.get_cls(ClassType.TRAINER)
                hps = HpoPipeStep._decode_hps(hps)
                trainer = cls_trainer(hps=hps, id=hpo_id, load_checkpoin=True)
                self.master.run(trainer)
            finished_trainer_info = self.master.pop_finished_worker()
            use_evaluator, cls_evaluator_set = self._use_evaluator()
            if use_evaluator:
                self._do_evaluate(finished_trainer_info, cls_evaluator_set)
                finished_trainer_info = self.master.pop_finished_evaluate_worker()
            self._update_hpo(finished_trainer_info)
        self.master.join()
        finished_trainer_info = self.master.pop_all_finished_train_worker()
        use_evaluator, cls_evaluator_set = self._use_evaluator()
        if use_evaluator:
            self._do_evaluate(finished_trainer_info, cls_evaluator_set)
            self.master.join()
            finished_trainer_info = self.master.pop_finished_evaluate_worker()
        self._update_hpo(finished_trainer_info)
        self.hpo._save_best()
        self._backup_output_path()
        self.master.close_client()

    def _use_evaluator(self):
        """Check if use evaluator and get the evaluators.

        :return: if we used evaluator, and Evaluator classes
        :rtype: bool, (Evaluator, GpuEvaluator, DloopEvaluator)
        """
        use_evaluator = False
        cls_evaluator = ClassFactory.get_cls(ClassType.EVALUATOR)
        cls_gpu_evaluator = None
        cls_dloop_evaluator = None
        try:
            cls_gpu_evaluator = ClassFactory.get_cls(ClassType.GPU_EVALUATOR)
            use_evaluator = True
        except Exception as e:
            logger.warning("evaluator not been set. {}".format(str(e)))
        try:
            cls_dloop_evaluator = ClassFactory.get_cls(ClassType.HAVA_D_EVALUATOR)
            use_evaluator = True
        except Exception:
            pass
        return use_evaluator, (cls_evaluator, cls_gpu_evaluator, cls_dloop_evaluator)

    def _do_evaluate(self, finished_trainer_info, cls_evaluator_set):
        """Do evaluate stuff.

        :param finished_trainer_info: the finished trainer info
        :type: list or dict
        :param cls_evaluator_set: set of evaluator classes.
        :type: set, (Evaluator, GpuEvaluator, DloopEvaluator)

        """
        if not isinstance(finished_trainer_info, list):
            finished_trainer_info = [finished_trainer_info]
        for worker_info in finished_trainer_info:
            cls_evaluator, cls_gpu_evaluator, cls_dloop_evaluator = cls_evaluator_set
            if cls_evaluator is None:
                break
            evaluator = cls_evaluator(worker_info=worker_info)
            if cls_gpu_evaluator is not None and worker_info is not None:
                evaluator.add_evaluator(cls_gpu_evaluator(worker_info=worker_info, load_checkpoint=True))
            if cls_dloop_evaluator is not None and worker_info is not None:
                evaluator.add_evaluator(cls_dloop_evaluator(worker_info=worker_info))
            if evaluator.size > 0:
                self.master.run(worker=evaluator)

    @staticmethod
    def _decode_hps(hps):
        """Decode hps: `trainer.optim.lr : 0.1` to dict format.

        And convert to `vega.core.common.config import Config` object
        This Config will be override in Trainer or Datasets class
        The override priority is: input hps > user configuration >  default configuration
        :param hps: hyper params
        :return: dict
        """
        hps_dict = {}
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

    def _update_hpo(self, worker_info):
        """Get finished worker's info, and use it to update target `generator`.

        Will get the finished worker's working dir, and then call the function
        `generator.update(worker_result_path)`.
        :param worker_info: `worker_info` is the finished worker's info, usually
            a dict or list of dict include `step_name` and `worker_id`.
        :type worker_info: dict or list of dict

        """
        if worker_info is None:
            return
        if not isinstance(worker_info, list):
            worker_info = [worker_info]
        for one_info in worker_info:
            step_name = one_info["step_name"]
            worker_id = one_info["worker_id"]
            logger.info("update hpo, step name: {}, worker id: {}".format(step_name, worker_id))
            try:
                self.hpo.update(step_name, worker_id)
            except Exception:
                logger.error("Failed to HPO update, step_name={}, worker_id={}.".format(step_name, worker_id))
                logger.error(traceback.format_exc())
