# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer."""

import zeus
from zeus.trainer.trainer_base import TrainerBase


class Trainer(TrainerBase):
    """Trainer class."""

    def __new__(cls, model=None, id=None, hps=None, load_ckpt_flag=False,
                model_desc=None, lazy_build=True, **kwargs):
        """Create Trainer clss."""
        if zeus.is_torch_backend():
            from zeus.trainer.trainer_torch import TrainerTorch
            trainer_cls = TrainerTorch
        elif zeus.is_tf_backend():
            from zeus.trainer.trainer_tf import TrainerTf
            trainer_cls = TrainerTf
        else:
            from zeus.trainer.trainer_ms import TrainerMs
            trainer_cls = TrainerMs

        return trainer_cls(model=model, id=id, hps=hps, load_ckpt_flag=load_ckpt_flag,
                           model_desc=model_desc, lazy_build=lazy_build, **kwargs)
