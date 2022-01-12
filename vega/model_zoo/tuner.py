# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base Trainer."""

import logging

import vega
from vega.common import FileOps, TaskOps
from vega.core.pipeline.conf import PipeStepConfig
from vega.model_zoo import ModelZoo
from vega.report import ReportClient
from vega.common.class_factory import ClassFactory, ClassType


class ModelTuner(object):
    """Model Tuner that can call the nas algorithm to search a new model."""

    __worker_id__ = None
    __step_name__ = None
    __fns__ = None

    @classmethod
    def setup(cls, step_name, worker_id):
        """Set step name and work id."""
        cls.__step_name__ = step_name
        cls.__worker_id__ = worker_id

    @classmethod
    def register_fn(cls, fn_name, **kwargs):
        """Register function and params."""
        cls.__fns__ = [fn_name, kwargs]

    @classmethod
    def get_fn(cls):
        """Get function info."""
        return tuple(cls.__fns__)

    @classmethod
    def build_model(cls, model):
        """Build a new mode by call dag search algorithm."""
        logging.info("Start tune model.")
        record = ReportClient().get_record(cls.__step_name__, cls.__worker_id__)
        device = next(model.parameters()).device
        if not record or not record.desc:
            model = cls.build_on_fine_tune(model)
        model = cls.build_after_nas(model)
        return model.to(device)

    @classmethod
    def build_on_fine_tune(cls, model):
        """Parse model to desc on the first time."""
        step_name, worker_id = cls.__step_name__, cls.__worker_id__
        dag_cls = ClassFactory.get_cls(ClassType.NETWORK, 'Script2Vega')
        dag_model = dag_cls(model=model)()
        desc = dag_model.to_desc()
        ReportClient().update(step_name, worker_id, desc=desc)
        cls._save(dag_model)
        logging.info("End to Fine tune model.")
        return model

    @classmethod
    def build_after_nas(cls, model):
        """Build a new model on nas pipe step."""
        record = ReportClient().get_record(cls.__step_name__, cls.__worker_id__)
        dag_model = ModelZoo().get_model(record.desc, record.weights_file or PipeStepConfig.model.pretrained_model_file)
        ModelZoo().refine(model, dag_model)
        logging.info("End to tune model.")
        return model

    @classmethod
    def update(cls, model=None, performance=None):
        """Update performance and save weights."""
        ReportClient().update(cls.__step_name__, cls.__worker_id__, performance=performance)
        if model:
            cls._save(model)

    @classmethod
    def _save(cls, model):
        # save fine_tune weights.
        step_name, worker_id = cls.__step_name__, cls.__worker_id__
        weights_file = FileOps.join_path(
            TaskOps().get_local_worker_path(step_name, worker_id), "model_{}.pth".format(worker_id))
        if vega.is_torch_backend():
            import torch
            state_dict = model.state_dict()
            if isinstance(model, torch.nn.DataParallel):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            torch.save(state_dict, weights_file)
        elif vega.is_ms_backend():
            from mindspore.train.serialization import save_checkpoint
            save_checkpoint(model, weights_file)
