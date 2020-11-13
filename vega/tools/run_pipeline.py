# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run example."""

import os
import sys
import argparse
import vega
from copy import deepcopy
from zeus.common.general import General
from zeus.common.config import Config
from zeus.common import update_dict


def _append_env():
    dir_path = os.getcwd()
    sys.path.insert(0, dir_path)
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = dir_path
    else:
        os.environ["PYTHONPATH"] += ":{}".format(dir_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Vega")
    parser.add_argument("config_file", default=None, type=str,
                        help="Pipeline config file name")
    group_backend = parser.add_argument_group(title="Set backend and device, default is pytorch and GPU")
    group_backend.add_argument("-b", "--backend", default="pytorch", type=str,
                               choices=["pytorch", "p", "tensorflow", "t", "mindspore", "m"])
    group_backend.add_argument("-d", "--device", default="GPU", type=str,
                               choices=["GPU", "NPU"])
    group_resume = parser.add_argument_group(title="Resume not finished task")
    group_resume.add_argument("-r", "--resume", action='store_true',
                              help="Resume not finished task.")
    group_resume.add_argument("-t", "--task_id", default=None, type=str,
                              help="Specify the ID of the task to be resumed.")
    group_type = parser.add_argument_group(title='Choose startup method')
    group_type.add_argument("-s", "--startup", default='example', type=str,
                            choices=['example', 'e', 'benchmark', 'b'])
    group_config = parser.add_argument_group(title='Modify config for yml')
    group_config.add_argument("-m", "--modify", action='store_true',
                              help="Modify some config")
    group_config.add_argument("-dt", "--dataset", default=None, type=str,
                              help='Modify dataset for all pipe_step')
    group_config.add_argument("-dp", "--data_path", default=None, type=str,
                              help="Modify data_path for all pipe_step")
    group_config.add_argument("-bs", "--batch_size", default=None, type=str,
                              help='Modify batch_size of dataset for all pipe_step')
    group_config.add_argument("-es", "--epochs", default=None, type=str,
                              help='Modify fully_train epochs')
    args = parser.parse_args()
    return args


def _set_startup(args):
    if args.startup in ['benchmark', 'b']:
        cfg = Config(args.config_file)
        config = deepcopy(cfg)
        if 'benchmark' in cfg.keys():
            benchmark_config = cfg.pop('benchmark')
            config = update_dict(benchmark_config, cfg)
    else:
        config = Config(args.config_file)
    return config


def _modify_config(args, cfg):
    if isinstance(cfg, dict):
        for key in cfg.keys():
            if key in args.keys():
                if isinstance(cfg[key], dict):
                    cfg[key] = _modify_config(args[key], cfg[key])
                else:
                    cfg[key] = args[key]
            cfg[key] = _modify_config(args, cfg[key])
    return deepcopy(cfg)


def _check_parse(args):
    keys = [key for key in args.keys()]
    for key in keys:
        if args[key] is None:
            args.pop(key)
    if 'dataset' in args.keys():
        dataset_type = args['dataset']
        args['dataset'] = {'type': dataset_type}
    return args


def _set_backend(args):
    if args.backend in ["pytorch", "p"]:
        vega.set_backend("pytorch", args.device)
    elif args.backend in ["tensorflow", "t"]:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
        vega.set_backend("tensorflow", args.device)
    elif args.backend in ["mindspore", "m"]:
        vega.set_backend("mindspore", args.device)


def _resume(args):
    if args.resume:
        if not args.task_id:
            raise Exception("Please set task id (-t task_id) if you want resume not finished task.")
        from zeus.common.general import TaskConfig
        General.task.task_id = args.task_id
        General._resume = True
        TaskConfig.backup_original_value(force=True)
        General.backup_original_value(force=True)


def run_pipeline(load_special_lib_func=None):
    """Run pipeline."""
    args = _parse_args()
    _resume(args)
    _set_backend(args)
    _append_env()
    if load_special_lib_func:
        load_special_lib_func(args.config_file)
    config = _set_startup(args)
    args = vars(args)
    args = _check_parse(args)
    config = _modify_config(args, config)
    vega.run(config)


if __name__ == '__main__':
    run_pipeline()
