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
