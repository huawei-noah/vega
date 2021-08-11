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
import vega
from copy import deepcopy
from vega.common.general import General
from vega.common.config import Config
from vega.common.utils import verify_requires
from vega.common import argment_parser


def _append_env():
    dir_path = os.getcwd()
    sys.path.insert(0, dir_path)
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = dir_path
    else:
        os.environ["PYTHONPATH"] += ":{}".format(dir_path)


def _parse_args():
    parser = argment_parser("Run Vega")
    parser.add_argument("config_file", default=None, type=str,
                        help="Pipeline config file name")
    group_backend = parser.add_argument_group(
        title="set backend and device, priority: specified in the command line > "
        "specified in the configuration file > default settings(pytorch and GPU)")
    group_backend.add_argument("-b", "--backend", default=None, type=str,
                               choices=["pytorch", "p", "tensorflow", "t", "mindspore", "m"],
                               help="set training platform")
    group_backend.add_argument("-d", "--device", default=None, type=str,
                               choices=["GPU", "NPU"],
                               help="set training device")
    group_resume = parser.add_argument_group(title="Resume not finished task")
    group_resume.add_argument("-r", "--resume", action='store_true',
                              help="resume not finished task")
    group_resume.add_argument("-t", "--task_id", default=None, type=str,
                              help="specify the ID of the task to be resumed")
    group_config = parser.add_argument_group(title='Modify config for yml')
    group_config.add_argument("-m", "--modify", action='store_true',
                              help="modify some config")
    group_config.add_argument("-dt", "--dataset", default=None, type=str,
                              help='modify dataset for all pipe_step')
    group_config.add_argument("-dp", "--data_path", default=None, type=str,
                              help="modify data_path for all pipe_step")
    group_config.add_argument("-bs", "--batch_size", default=None, type=str,
                              help='modify batch_size of dataset for all pipe_step')
    group_config.add_argument("-es", "--epochs", default=None, type=str,
                              help='modify fully_train epochs')
    args = parser.parse_args()
    return args


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
    backend = args.backend
    device = args.device
    if backend:
        if args.backend in ["pytorch", "p"]:
            backend = "pytorch"
        elif args.backend in ["tensorflow", "t"]:
            backend = "tensorflow"
        elif args.backend in ["mindspore", "m"]:
            backend = "mindspore"
    else:
        config = Config(args.config_file)
        if "general" in config and "backend" in config["general"]:
            backend = config["general"]["backend"]
    if not device:
        config = Config(args.config_file)
        if "general" in config and "device_category" in config["general"]:
            device = config["general"]["device_category"]
    if backend:
        General.backend = backend
    if device:
        General.device_category = device
    vega.set_backend(General.backend, General.device_category)


def _resume(args):
    if args.resume:
        if not args.task_id:
            raise Exception("Please set task id (-t task_id) if you want resume not finished task.")
        from vega.common.general import TaskConfig
        General.task.task_id = args.task_id
        General._resume = True
        TaskConfig.backup_original_value(force=True)
        General.backup_original_value(force=True)


def _backup_config(args):
    _file = args.config_file
    from vega.common.task_ops import TaskOps
    from vega.common.file_ops import FileOps
    dest_file = FileOps.join_path(TaskOps().local_output_path, os.path.basename(_file))
    FileOps.make_base_dir(dest_file)
    FileOps.copy_file(_file, dest_file)


def _change_process_name():
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(bytes("vega-main", "utf-8"))
    libc.prctl(15, byref(buff), 0, 0, 0)


def run_pipeline(load_special_lib_func=None):
    """Run pipeline."""
    args = _parse_args()
    _resume(args)
    _set_backend(args)
    _append_env()
    if load_special_lib_func:
        load_special_lib_func(args.config_file)
    config = Config(args.config_file)
    # load general
    if config.get("general"):
        General.from_dict(config.get("general"), skip_check=False)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(General.TF_CPP_MIN_LOG_LEVEL)
    if General.requires and not verify_requires(General.requires):
        return
    dict_args = vars(args)
    dict_args = _check_parse(dict_args)
    config = _modify_config(dict_args, config)
    # _backup_config(args)
    _change_process_name()
    vega.run(config)


if __name__ == '__main__':
    run_pipeline()
