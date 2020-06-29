# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Benchmark Pipe Step defined in Pipeline."""
import logging
import glob
import os
import numpy as np
import traceback
from copy import deepcopy
from vega.core.common import TaskOps, Config, UserConfig, FileOps
from .pipe_step import PipeStep
from vega.core.common.class_factory import ClassFactory, ClassType


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class BenchmarkPipeStep(PipeStep):
    """Run pipeStep which is the base components class that can be added in Pipeline."""

    def __init__(self):
        super().__init__()

    def do(self):
        """Start to run benchmark evaluator."""
        logger.info("BenchmarkPipeStep started...")
        cfg = Config(deepcopy(UserConfig().data))
        step_name = cfg.general.step_name
        pipe_step_cfg = cfg[step_name].pipe_step
        if "esr_models_file" in pipe_step_cfg and pipe_step_cfg.esr_models_file is not None:
            # TODO: ESR model
            self._evaluate_esr_models(pipe_step_cfg.esr_models_file, pipe_step_cfg.models_folder)
        elif "models_folder" in pipe_step_cfg and pipe_step_cfg.models_folder is not None:
            self._evaluate_multi_models(pipe_step_cfg.models_folder)
        else:
            self._evaluate_single_model()
        self._backup_output_path()
        logger.info("Complete model evaluation.")

    def _evaluate_single_model(self, id=None, desc_file=None, pretrained_model=None):
        try:
            cls_gpu_evaluator = ClassFactory.get_cls(ClassType.GPU_EVALUATOR)
        except Exception:
            logger.error("Failed to create Evaluator, please check the config file.")
            logger.error(traceback.format_exc())
            return
        if desc_file and pretrained_model is not None:
            cls_gpu_evaluator.cfg.model_desc_file = desc_file
            model_cfg = ClassFactory.__configs__.get('model')
            if model_cfg:
                setattr(model_cfg, 'model_desc_file', desc_file)
            else:
                setattr(ClassFactory.__configs__, 'model', Config({'model_desc_file': desc_file}))
            cls_gpu_evaluator.cfg.pretrained_model_file = pretrained_model
        try:
            evaluator = cls_gpu_evaluator()
            evaluator.train_process()
            evaluator.output_evaluate_result(id, evaluator.evaluate_result)
        except Exception:
            logger.error("Failed to evaluate model, id={}, desc_file={}, pretrained_model={}".format(
                id, desc_file, pretrained_model))
            logger.error(traceback.format_exc())
            return

    def _evaluate_multi_models(self, models_folder):
        models_folder = models_folder.replace("{local_base_path}", self.task.local_base_path)
        models_folder = os.path.abspath(models_folder)
        model_desc_files = glob.glob("{}/model_desc_*.json".format(models_folder))
        pretrained_models = glob.glob("{}/model_*.pth".format(models_folder))
        for desc_file in model_desc_files:
            basename = os.path.basename(desc_file).replace(".json", ".pth").replace("model_desc_", "model_")
            pretrained_model = os.path.join(os.path.dirname(desc_file), basename)
            if pretrained_model not in pretrained_models:
                logger.warn("No pretrained model corresponding to the model desc file, file={}".format(desc_file))
                continue
            id = os.path.splitext(os.path.basename(desc_file))[0][11:]
            logger.info("Begin evaluate model, id={}, desc={}".format(id, desc_file))
            self._evaluate_single_model(id, desc_file, pretrained_model)

    def _evaluate_esr_models(self, esr_models_file, models_folder):
        models_folder = models_folder.replace("{local_base_path}", self.task.local_base_path)
        models_folder = os.path.abspath(models_folder)
        esr_models_file = esr_models_file.replace("{local_base_path}", self.task.local_base_path)
        esr_models_file = os.path.abspath(esr_models_file)
        archs = np.load(esr_models_file)
        for i, arch in enumerate(archs):
            try:
                cls_gpu_evaluator = ClassFactory.get_cls(ClassType.GPU_EVALUATOR)
            except Exception:
                logger.error("Failed to create Evaluator, please check the config file")
                logger.error(traceback.format_exc())
                return
            pretrained_model = FileOps.join_path(models_folder, "model_{}.pth".format(i))
            if not os.path.exists(pretrained_model):
                logger.error("Failed to find model file, file={}".format(pretrained_model))
            cls_gpu_evaluator.cfg.model_arch = arch
            cls_gpu_evaluator.cfg.pretrained_model_file = pretrained_model
            try:
                evaluator = cls_gpu_evaluator()
                evaluator.train_process()
                evaluator.output_evaluate_result(i, evaluator.evaluate_result)
            except Exception:
                logger.error("Failed to evaluate model, id={}, pretrained_model={}".format(i, pretrained_model))
                logger.error(traceback.format_exc())
                return
