# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Utils for finding module."""

from __future__ import division, print_function

import os
import argparse
import sys
import ast
import socket
import yaml


def import_config(global_para, config):
    """
    Import config.

    :param global_para
    :param config
    :return: None
    """
    if not config:
        return
    for key in config.keys():
        if key in global_para:
            global_para[key] = config[key]


def node_para(args):
    """
    Node parameters.

    :param args: args
    :return: node config or test node config list
    """
    node_list = []
    i = 0
    if args.find("//") >= 0:
        for node in args.split("//"):
            node_list.append([])
            ip, name, passwd = node.split(",")
            node_list[i].append(ip)
            node_list[i].append(name)
            node_list[i].append(passwd)
            i += 1
    else:
        node_list.append([])
        ip, name, passwd = args.split(",")
        node_list[i].append(ip)
        node_list[i].append(name)
        node_list[i].append(passwd)
    return node_list


def analysis_para(args):
    """
    Analysis parameters.

    :param args:
    :return: Dictionary of args
    """
    dict_args = {}
    for kv in args.split(","):
        key, value = kv.split("=")
        if key == "action_dim":
            value = int(value)
            dict_args[key] = value
        elif key == "state_dim":
            value = ast.literal_eval(value)
            dict_args[key] = value
        elif key == "vision":
            if value == "True":
                value = True
            else:
                value = False
            dict_args[key] = value
        else:
            dict_args[key] = value
    return dict_args


def get_config_file():
    """
    Get config file.

    :return: config file for training or testing
    """
    parser = argparse.ArgumentParser(
        description="parse key pairs into a dictionary for xt training or testing",
        usage="python train.py --config_file YAML_FILE OR "
        "python train.py --alg_para KEY=VALUE "
        "--env_para KEY=VALUE --env_info KEY=VALUE --agent_para KEY=VALUE "
        "--actor KEY=VALUE",
    )

    parser.add_argument("-f", "--config_file")
    parser.add_argument(
        "-s3", "--save_to_s3", default=None, help="save model into s3 bucket."
    )

    parser.add_argument("--alg_para", type=analysis_para)
    parser.add_argument("--alg_config", type=analysis_para)

    parser.add_argument("--env_para", type=analysis_para)
    parser.add_argument("--env_info", type=analysis_para)

    parser.add_argument("--agent_para", type=analysis_para)
    parser.add_argument("--agent_config", type=analysis_para)

    parser.add_argument("--actor", type=analysis_para)
    parser.add_argument("--critic", type=analysis_para)

    parser.add_argument("--model_name", default="model_name")
    parser.add_argument("--env_num", type=int, default=1)

    parser.add_argument(
        "--node_config", type=node_para, default=[["127.0.0.1", "username", "passwd"]]
    )
    parser.add_argument("--test_node_config", type=node_para)

    # parser.add_argument(
    #     "--model_path", default="../xt_train_data/test_model/" + str(os.getpid())
    # )
    parser.add_argument(
        "--test_model_path", default="../xt_train_data/train_model/" + str(os.getpid())
    )
    parser.add_argument(
        "--result_path",
        default="../xt_train_data/test_res/" + str(os.getpid()) + ".csv",
    )

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        print(parser.print_help())
        exit(1)
    if args.config_file is not None:
        return args.config_file, args.save_to_s3
    args_dict = vars(args)

    model_para = {}
    model_para["actor"] = args_dict["actor"]
    model_para["critic"] = args_dict["critic"]
    args_dict["model_para"] = model_para

    args_dict.pop("actor")
    args_dict.pop("critic")

    args_dict["env_para"]["env_info"] = args_dict["env_info"]
    args_dict.pop("env_info")

    if args_dict["agent_config"] is not None:
        args_dict["agent_para"]["agent_config"] = args_dict["agent_config"]
    args_dict.pop("agent_config")

    if args_dict["alg_config"] is not None:
        args_dict["alg_para"]["alg_config"] = args_dict["alg_config"]
    args_dict.pop("alg_config")

    if args_dict["test_node_config"] is None:
        args_dict.pop("test_node_config")

    yaml_file = "./xt_{}_{}.yaml".format(
        args_dict["alg_para"]["alg_name"], args_dict["env_para"]["env_info"]["name"]
    )
    with open(yaml_file, "w") as f:
        f.write(yaml.dump(args_dict))

    return yaml_file, args.save_to_s3


def check_port(ip, port):
    """Check if port  is in use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        print("port is used", int(port))
        return True
    except BaseException:
        return False


def bytes_to_str(data):
    """Bytes to string, used after data transform by internet."""
    if isinstance(data, bytes):
        return data if sys.version_info.major == 2 else data.decode("ascii")

    if isinstance(data, dict):
        return dict(map(bytes_to_str, data.items()))

    if isinstance(data, tuple):
        return map(bytes_to_str, data)

    return data


def get_host_ip():
    """Get local ip address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip
