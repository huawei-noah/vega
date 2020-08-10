# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Conf for Pipeline."""
from vega.core.common.class_factory import ClassType
from vega.core.trainer.conf import TrainerConfig
from vega.core.evaluator.conf import EvaluatorConfig


class SearchSpaceConfig(object):
    """Default Search Space config for Pipeline."""

    _type_name = ClassType.SEARCH_SPACE
    type = None


class ModelConfig(object):
    """Default Model config for Pipeline."""

    _type_name = ClassType.SEARCH_SPACE
    type = None
    model_desc = None
    model_desc_file = None
    model_file = None


class SearchAlgorithmConfig(object):
    """Default Search Algorithm config for Pipeline."""

    _class_type = ClassType.SEARCH_ALGORITHM
    type = None


class DatasetConfig(object):
    """Default Dataset config for Pipeline."""

    type = "Cifar10"
    _class_type = ClassType.DATASET


class PipeStepConfig(object):
    """Default Pipeline config for Pipe Step."""

    dataset = DatasetConfig
    search_algorithm = SearchAlgorithmConfig
    search_space = SearchSpaceConfig
    model = ModelConfig
    trainer = TrainerConfig
    evaluator = EvaluatorConfig
    pipe_step = {}


class PipelineConfig(object):
    """Pipeline config."""

    steps = []
