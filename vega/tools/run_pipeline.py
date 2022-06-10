# -*- coding:utf-8 -*-

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

"""Run pipeline."""

import os
import sys
from copy import deepcopy

import vega
from vega.common.general import General
from vega.common.config import Config
from vega.common.utils import verify_requires, verify_platform_pkgs
from vega.common.arg_parser import argment_parser, str2bool
from vega import security


def _append_env():
    dir_path = os.getcwd()
    sys.path.insert(0, dir_path)
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = dir_path
    elif dir_path not in os.environ["PYTHONPATH"].split(":"):
        os.environ["PYTHONPATH"] += f":{dir_path}"


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
    group_config = parser.add_argument_group(title='Modify default configs in yml')
    group_config.add_argument("-dt", "--dataset", default=None, type=str,
                              help='modify dataset for all pipe_step')
    group_config.add_argument("-dp", "--data_path", default=None, type=str,
                              help="modify data_path for all pipe_step")
    group_config.add_argument("-bs", "--batch_size", default=None, type=int,
                              help='modify batch_size of dataset for all pipe_step')
    group_config.add_argument("-es", "--epochs", default=None, type=int,
                              help='modify fully_train epochs')
    group_cluster = parser.add_argument_group(title='Set cluster info')
    group_cluster.add_argument("-sa", "--standalone_boot", default=None, type=str2bool,
                               help="standalone boot mode, eg. -sa true")
    group_cluster.add_argument("-ps", "--parallel_search", default=None, type=str2bool,
                               help="parallel search")
    group_cluster.add_argument("-pt", "--parallel_fully_train", default=None, type=str2bool,
                               help="parallel fully train")
    group_cluster.add_argument("-mi", "--master_ip", default=None, type=str,
                               help="master ip, eg. -mi n.n.n.n")
    group_cluster.add_argument("-ws", "--num_workers", default=None, type=int,
                               help="number of workers, eg. -ws 12")
    group_cluster.add_argument("-p", "--listen_port", default=None, type=int,
                               help="listen port, eg. -p 8878")
    group_cluster.add_argument("-sv", "--slaves", dest="slaves", nargs="+",
                               help="slaves, eg. -sv n.n.n.n n.n.n.n")
    parser = security.add_args(parser)
    args = parser.parse_args()
    if args.security:
        security.check_args(args)
        security.check_yml(args.config_file)
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


def _get_backend_device(args):
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
    return General.backend, General.device_category


def _resume(args):
    if args.resume:
        if not args.task_id:
            raise Exception("Please set task id (-t task_id) if you want resume not finished task.")
        from vega.common.general import TaskConfig
        General.task.task_id = args.task_id
        General._resume = True
        TaskConfig.backup_original_value(force=True)
        General.backup_original_value(force=True)


def _backup_config(file_name, config):
    from vega.common import FileOps, TaskOps
    config.dump_yaml(FileOps.join_path(TaskOps().local_output_path, os.path.basename(file_name)))


def _change_process_name():
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(bytes("vega-main", "utf-8"))
    libc.prctl(15, byref(buff), 0, 0, 0)


def _set_cluster(args, config):
    if "general" not in config:
        config["general"] = {}
    if "cluster" not in config["general"]:
        config["general"]["cluster"] = {}
    for key in ["parallel_search", "parallel_fully_train"]:
        if args.get(key, None) is not None:
            setattr(General, key, args.get(key))
            config["general"][key] = args.get(key)
    for key in ["standalone_boot", "num_workers", "master_ip", "listen_port", "slaves"]:
        if args.get(key, None) is not None:
            setattr(General.cluster, key, args.get(key))
            config["general"]["cluster"][key] = args.get(key)
    return config


def _check_platform_pkgs(backend, device):
    result = True
    if backend == "pytorch":
        result = verify_platform_pkgs([
            ("torch", "torch"),
            ("torchvision", "torchvision")])
    elif backend == "tensorflow":
        if device == "GPU":
            tensorflow = "tensorflow-gpu>=1.14.0,<2.0"
        else:
            tensorflow = "tensorflow"
        result = verify_platform_pkgs([
            ("tensorflow", tensorflow),
            ("tf_slim", "tf-slim"),
            ("official", "tf-models-official==0.0.3.dev1")])
    elif backend == "mindspore":
        result = verify_platform_pkgs([
            ("mindspore", "mindspore")])
    return result


def main():
    """Run pipeline."""
    try:
        args = _parse_args()
    except Exception as e:
        print(f"Parameter Error: {e}")
        return
    _resume(args)
    if args.security:
        os.umask(0o077)
        if not security.load_config("all"):
            print("If you want to run vega in normal mode, do not use parameter '-s'.")
            print("For more parameters: vega --help")
            return
    General.security = args.security
    (backend, device) = _get_backend_device(args)
    if not _check_platform_pkgs(backend, device):
        return
    vega.set_backend(backend, device)
    _append_env()
    config = Config(args.config_file, abs_path=True)
    if General.security:
        try:
            if not security.check_risky_files([args.config_file]):
                return
            if not security.check_risky_file_in_config(args, config):
                return
            if not security.check_env(config):
                return
        except Exception as e:
            print(f"Secrity Error: {e}")
            return
    # load general
    if config.get("general"):
        General.from_dict(config.get("general"), skip_check=False)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(General.TF_CPP_MIN_LOG_LEVEL)
    if General.requires and not verify_requires(General.requires):
        return
    dict_args = vars(args)
    dict_args = _check_parse(dict_args)
    config = _set_cluster(dict_args, config)
    config = _modify_config(dict_args, config)
    _backup_config(args.config_file, config)
    _change_process_name()
    vega.run(config)


if __name__ == '__main__':
    main()
