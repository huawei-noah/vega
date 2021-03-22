# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register trainer automatically."""


def register_trainer(backend):
    """Import and register trainer automatically."""
    if backend == "pytorch":
        from zeus.trainer.trainer_torch import TrainerTorch
    elif backend == "tensorflow":
        from zeus.trainer.trainer_tf import TrainerTf
    elif backend == "mindspore":
        from zeus.trainer.trainer_ms import TrainerMs
