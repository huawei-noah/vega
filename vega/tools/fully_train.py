# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""
import vega
from vega.core.pipeline.fully_train_pipe_step import FullyTrainPipeStep
from vega.tools.init_env import _init_env
from vega.tools.args import _parse_args, _set_config


def _fully_train():
    args = _parse_args(["cluster", "model", "trainer"], "Fully train model.")
    vega.set_backend(args.general.backend)
    _set_config(args, "fully_train", "FullyTrainPipeStep")
    _init_env()
    FullyTrainPipeStep().do()


if __name__ == "__main__":
    _fully_train()
