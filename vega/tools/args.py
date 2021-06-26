# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""
from vega.common import argment_parser
from vega.common.config import Config, build_tree
from vega.common.general import General
from vega.core.pipeline.conf import PipelineConfig, PipeStepConfig
from vega.trainer.conf import TrainerConfig


def _parse_args(sections, desc):
    parser = argment_parser(desc)
    parser.add_argument("-backend", "--general.backend", default="pytorch", type=str,
                        help="pytorch|tensorflow|mindspore")
    if "cluster" in sections:
        parser.add_argument("-devices_per_trainer", "--general.worker.devices_per_trainer", default=None, type=int)
        parser.add_argument("-master_ip", "--general.cluster.master_ip", default=None, type=str)
        parser.add_argument("-slaves", "--general.cluster.slaves", default=[],
                            action='store', dest='general.cluster.slaves', type=str, nargs='*',
                            help="slave IP list")
    parser.add_argument("-dataset", "--dataset.type", required=True, type=str, help="dataset name.")
    parser.add_argument("-data_path", "--dataset.common.data_path", type=str, help="dataset path.")
    parser.add_argument("-batch_size", "--dataset.common.batch_size", default=256, type=int)
    if "model" in sections:
        parser.add_argument("-model_desc", "--model.model_desc", type=str)
        parser.add_argument("-model_file", "--model.pretrained_model_file", type=str)
    if "trainer" in sections:
        parser.add_argument("-epochs", "--trainer.epochs", type=int)
    if "fine_tune" in sections:
        parser.add_argument("-task_type", "--task_type", default="classification", type=str,
                            help="classification|detection|segmentation|super_resolution")
        parser.add_argument("-num_classes", "--trainer.num_classes", type=int)
    parser.add_argument("-evaluator", "--evaluator", default=[],
                        action='store', dest='evaluator', type=str, nargs='*',
                        help="evaluator list, eg. -evaluator HostEvaluator DeviceEvaluator")
    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if args[key]}
    tree = Config(build_tree(args))
    return tree


def _set_config(args, step_name, step_type):
    """Fully train."""
    # general
    General.step_name = step_name
    if hasattr(args, "general"):
        General.from_dict(args.general)
    # pipeline
    PipelineConfig.steps = [step_name]
    # pipestep
    PipeStepConfig.type = step_type
    # model
    if hasattr(args, "model"):
        if hasattr(args.model, "model_desc"):
            args.model.model_desc = Config(args.model.model_desc)
        PipeStepConfig.model.from_dict(args.model)
    # dataset
    if hasattr(args, "dataset"):
        PipeStepConfig.dataset.from_dict(args.dataset)
    # trainer
    if hasattr(args, "trainer"):
        TrainerConfig.from_dict(args.trainer)
    # evaluator
    if hasattr(args, "evaluator"):
        # PipeStepConfig.evaluator._type_name = args.evaluator
        if "HostEvaluator" in args.evaluator:
            PipeStepConfig.evaluator_enable = True
            PipeStepConfig.evaluator.host_evaluator_enable = True
        if "DeviceEvaluator" in args.evaluator:
            PipeStepConfig.evaluator_enable = True
            PipeStepConfig.evaluator.device_evaluator_enable = True
