# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer."""

import vega
from vega.trainer.trainer_base import TrainerBase
from vega.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRAINER)
class Trainer(TrainerBase):
    """Trainer class."""

    def __new__(cls, model=None, id=None, hps=None, load_ckpt_flag=False,
                model_desc=None, **kwargs):
        """Create Trainer clss."""
        if vega.is_torch_backend():
            trainer_cls = ClassFactory.get_cls(ClassType.TRAINER, "TrainerTorch")
        elif vega.is_tf_backend():
            trainer_cls = ClassFactory.get_cls(ClassType.TRAINER, "TrainerTf")
        else:
            trainer_cls = ClassFactory.get_cls(ClassType.TRAINER, "TrainerMs")
        return trainer_cls(model=model, id=id, hps=hps, load_ckpt_flag=load_ckpt_flag,
                           model_desc=model_desc, **kwargs)
