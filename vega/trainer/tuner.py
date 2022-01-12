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
from vega.common.class_factory import ClassFactory, ClassType
from vega.common.wrappers import train_process_wrapper
from vega.trainer.trainer_base import TrainerBase
from vega.model_zoo.tuner import ModelTuner


@ClassFactory.register(ClassType.TRAINER)
class Tuner(TrainerBase):
    """Tuner to call user function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @train_process_wrapper
    def train_process(self):
        """Define train process."""
        fn_name, fn_kwargs = ModelTuner.get_fn()
        ModelTuner.setup(self.step_name, self._worker_id)
        if hasattr(self.config, "params") and self.config.params:
            fn_kwargs.update(self.config.params)
        logging.info("function args: {}".format(fn_kwargs))
        return fn_name(**fn_kwargs)
