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

import logging
import os
import stat


def _file_exist(path):
    return os.access(path, os.F_OK)


def _file_belong_to_current_user(path):
    return os.stat(path).st_uid == os.getuid()


def _file_other_writable(path):
    return os.stat(path).st_mode & stat.S_IWOTH


def _file_is_link(path):
    return os.path.islink(path)


def _get_risky_files_by_suffix(suffixes, path):
    risky_files = []
    other_user_files = []
    writable_files = []
    links = []
    for suffix in suffixes:
        if not path.endswith(suffix):
            continue
        abs_path = os.path.abspath(path)
        if _file_exist(abs_path):
            risky_files.append(abs_path)
            if not _file_belong_to_current_user(abs_path):
                other_user_files.append(abs_path)
            if _file_other_writable(abs_path):
                writable_files.append(abs_path)
            if _file_is_link(abs_path):
                links.append(abs_path)

    return risky_files, other_user_files, writable_files, links


def _get_risky_files_in_config(config, risky_files, other_user_files, writable_files, links):
    """Get contained risky file (.pth/.pth.tar/.onnx/.py)."""
    for value in config.values():
        if isinstance(value, dict) and value.get("type", "") == "DeepLabNetWork":
            value = value.get("dir").rstrip("/") + "/" + value.get("name").lstrip("/") + ".py"
        if isinstance(value, str):
            _riskies, _others, _writables, _links = _get_risky_files_by_suffix([".pth", ".pth.tar", ".py"], value)
            risky_files.extend(_riskies)
            other_user_files.extend(_others)
            writable_files.extend(_writables)
            links.extend(_links)
        if isinstance(value, dict):
            _get_risky_files_in_config(value, risky_files, other_user_files, writable_files, links)


def check_risky_file_in_config(args, config):
    """Check risky file (.pth/.pth.tar/.py)."""
    if not args.security:
        return True
    risky_files = []
    other_user_files = []
    writable_files = []
    links = []
    _get_risky_files_in_config(config, risky_files, other_user_files, writable_files, links)
    if len(risky_files) == 0:
        return True

    print("\033[1;33m"
          "WARNING: The following executable files will be loaded:"
          "\033[0m")
    for file in risky_files:
        print(file)
    if len(other_user_files) > 0:
        print("\033[1;33m"
              "WARNING: The following executable files that will be loaded do not belong to the current user:"
              "\033[0m")
        for file in other_user_files:
            print(file)
    if len(writable_files) > 0:
        print("\033[1;33m"
              "WARNING: The following executable files that will be loaded have others write permission:"
              "\033[0m")
        for file in writable_files:
            print(file)
    if len(links) > 0:
        print("\033[1;33m"
              "WARNING: The following executable files that will be loaded is soft link file:"
              "\033[0m")
        for file in links:
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


def check_risky_files(file_list):
    """Check if cert and key file are risky."""
    res = True
    for file in file_list:
        if file != '' and not os.path.exists(file):
            logging.error(f"File <{file}> does not exist")
            res = False
            continue
        if not _file_belong_to_current_user(file):
            logging.error(f"File <{file}> is not owned by current user")
            res = False
        if _file_is_link(file):
            logging.error(f"File <{file}> should not be soft link")
            res = False
        if os.stat(file).st_mode & 0o0177:
            logging.error(f"File <{file}> permissions are not correct, cannot exceed 600")
            res = False
    return res
