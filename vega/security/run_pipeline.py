# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run pipeline."""
import configparser
import logging
import os
import stat
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
    group_config.add_argument("-f", "--force", default=None, action="store_true",
                              help='skip check validation of pretrained model')
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


class LoadConfigException(Exception):
    """Load config exception."""

    pass


def _read_config_file():
    """Read config file and return ConfigParser."""
    vega_config_file = os.path.join(os.environ['HOME'], ".vega", "vega.ini")
    if not os.path.exists(vega_config_file):
        raise LoadConfigException(f"Not found configure file: {vega_config_file}")
    config = configparser.ConfigParser()
    config.read(vega_config_file)
    return config


def _parse_config(config):
    General.security_setting = config._sections
    General.security_setting.get("security")["enable"] = True \
        if str(General.security_setting.get("security").get("enable")).upper() == "TRUE" else False


def _get_config_field(config, field):
    if field not in config:
        raise LoadConfigException("field <{}> is not existed in config file".format(field))
    return config[field]


def _get_config_key(config, key, field):
    if key not in config:
        raise LoadConfigException("key <{}> is not in field <{}> of config file".format(key, field))
    return config[key]


def _check_if_file_config_correct(config, key, field):
    file = _get_config_key(config, key, field)
    if not os.path.exists(file):
        raise LoadConfigException("file <{}> is not existed.".format(file))


def _check_security_switch_valid(config):
    if "security" not in config or "enable" not in config["security"]:
        raise LoadConfigException("Invalid config file: security field must be included")


def _get_security_switch_on_off(config):
    return True if config["security"]["enable"].upper() == "TRUE" else False


def load_security_setting():
    """Load security settings."""
    try:
        config = _read_config_file()
        _check_security_switch_valid(config)
        security_mode = _get_security_switch_on_off(config)
        if not security_mode:
            General.security_setting = {
                "security": {
                    "enable": False
                }
            }
            return True
        _check_config_validation(config)
        _parse_config(config)
    except LoadConfigException as e:
        logging.warning("load_security_setting failed: {}".format(e))
        return False
    return True


def _check_cert_key_file(config, key, field):
    file = _get_config_key(config, key, field)
    if not os.stat(file).st_uid == os.getuid():
        raise Exception("File <{}> is not owned by current user".format(file))
    if os.path.islink(file):
        raise Exception("File <{}> should not be soft link".format(file))
    if os.stat(file).st_mode & 0o0077:
        raise Exception("file <{}> is accessible by group/other users".format(file))


def _check_config_validation(config):
    https_config = _get_config_field(config, "https")
    _check_if_file_config_correct(https_config, "cert_pem_file", "https")
    _check_cert_key_file(https_config, "cert_pem_file", "https")


def _file_exist(path):
    return os.access(path, os.F_OK)


def _file_belong_to_current_user(path):
    return os.stat(path).st_uid == os.getuid()


def _file_other_writable(path):
    return os.stat(path).st_mode & stat.S_IWOTH


def _file_is_link(path):
    return os.path.islink(path)


def check_env():
    """Check environment."""
    if not load_security_setting():
        return False
    return True


def _get_risky_files_by_suffix(suffixes, path):
    risky_files = []
    non_current_user_files = []
    others_writable_files = []
    link_files = []
    for suffix in suffixes:
        if not path.endswith(suffix):
            continue
        abs_path = os.path.abspath(path)
        if _file_exist(abs_path):
            risky_files.append(abs_path)
            if not _file_belong_to_current_user(abs_path):
                non_current_user_files.append(abs_path)
            if _file_other_writable(abs_path):
                others_writable_files.append(abs_path)
            if _file_is_link(abs_path):
                link_files.append(abs_path)

    return risky_files, non_current_user_files, others_writable_files, link_files


def get_risky_files(config):
    """Get contained risky file (.pth/.pth.tar/.onnx/.py)."""
    risky_files = []
    non_current_user_files = []
    others_writable_files = []
    link_files = []

    if not isinstance(config, Config):
        return risky_files, non_current_user_files, others_writable_files, link_files

    for value in config.values():
        if isinstance(value, Config) and value.get("type") == "DeepLabNetWork":
            value = value.get("dir").rstrip("/") + "/" + value.get("name").lstrip("/") + ".py"
        if isinstance(value, str):
            temp_risky_files, temp_non_current_user_files, temp_other_writable_files, temp_link_files \
                = _get_risky_files_by_suffix([".pth", ".pth.tar", ".py"], value)
            risky_files.extend(temp_risky_files)
            non_current_user_files.extend(temp_non_current_user_files)
            others_writable_files.extend(temp_other_writable_files)
            link_files.extend(temp_link_files)
        temp_risky_files, temp_non_current_user_files, temp_other_writable_files, temp_link_files \
            = get_risky_files(value)
        risky_files.extend(temp_risky_files)
        non_current_user_files.extend(temp_non_current_user_files)
        others_writable_files.extend(temp_other_writable_files)
        link_files.extend(temp_link_files)

    return risky_files, non_current_user_files, others_writable_files, link_files


def check_risky_file(args, config):
    """Check risky file (.pth/.pth.tar/.py)."""
    if args.force:
        return True
    risky_files, non_current_user_files, others_writable_files, link_files = get_risky_files(config)
    if len(risky_files) == 0:
        return True

    print("\033[1;33m"
          "WARNING: The following executable files will be loaded:"
          "\033[0m")
    for file in risky_files:
        print(file)
    if len(non_current_user_files) > 0:
        print("\033[1;33m"
              "WARNING: The following executable files that will be loaded do not belong to the current user:"
              "\033[0m")
        for file in non_current_user_files:
            print(file)
    if len(others_writable_files) > 0:
        print("\033[1;33m"
              "WARNING: The following executable files that will be loaded have others write permission:"
              "\033[0m")
        for file in others_writable_files:
            print(file)
    if len(link_files) > 0:
        print("\033[1;33m"
              "WARNING: The following executable files that will be loaded is soft link file:"
              "\033[0m")
        for file in link_files:
            print(file)
    user_confirm = input("It is possible to construct malicious pickle data "
                         "which will execute arbitrary code during unpickling .pth/.pth.tar/.py files. "
                         "\nPlease ensure the safety and consistency of the loaded executable files. "
                         "\nDo you want to continue? (yes/no) ").strip(" ")
    while user_confirm != "yes" and user_confirm != "no":
        user_confirm = input("Please enter yes or no! ").strip(" ")
    if user_confirm == "yes":
        return True
    elif user_confirm == "no":
        return False


def run_pipeline(load_special_lib_func=None):
    """Run pipeline."""
    os.umask(0o027)
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
    # check env
    if not check_env():
        return
    if not check_risky_file(args, config):
        return
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
