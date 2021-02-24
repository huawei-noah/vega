# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fully Train PipeStep that used in Pipeline."""
import os
import logging
import subprocess
import pickle
import glob
import traceback
from copy import deepcopy
import numpy as np
from .pipe_step import PipeStep
from ..common.class_factory import ClassFactory, ClassType
from vega.core.common import UserConfig, FileOps, TaskOps, Config
from ..scheduler import Master, LocalMaster
from vega.search_space.networks.network_factory import NetworkFactory
from .fully_train_pipe_step import FullyTrainPipeStep
import torch
import numpy as np

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class FullyTrainPipeStepFixedSeed(FullyTrainPipeStep):
    """FullyTrainPipeStepMF is the implementation class of PipeStep for with fixed seed.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def __init__(self):
        super().__init__()
        logger.info("init FullyTrainPipeStepFixedSeed...")
        seed = self.cfg.get('seed', 99999)
        np.random.seed(seed)
        torch.manual_seed(seed)