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
import os
from argparse import ArgumentParser


def read_config_file():
    """Read config file and return ConfigParser."""
    vega_config_file = os.path.join(os.environ['HOME'], ".vega", "vega.ini")
    if not os.path.exists(vega_config_file):
        print(f"Not found vega security configure file: {vega_config_file}")
        return None
    config = configparser.ConfigParser()
    config.read(vega_config_file)
    return config


def _parse_args():
    parser = ArgumentParser("Vega Configuration")
    group_resume = parser.add_mutually_exclusive_group(required=True)
    group_resume.add_argument("-i", "--init", action='store_true',
                              help="init vega security config file")
    group_resume.add_argument("-q", "--query", type=str, choices=["sec", "https"],
                              help="query current vega security setting")
    group_resume.add_argument("-s", "--set", type=int, choices=[0, 1],
                              help="set vega security mode to be on or off")
    group_resume.add_argument("-m", "--module", type=str, choices=["https"],
                              help="set cert/key file of each module")

    group_config = parser.add_argument_group(title='cert key files')
    group_config.add_argument("-ca", "--ca-cert", default=None, type=str,
                              help="ca cert file")
    group_config.add_argument("-c", "--cert", default=None, type=str,
                              help='server cert file')
    group_config.add_argument("-p", "--public-key", default=None, type=str,
                              help="server public key file")
    group_config.add_argument("-k", "--secret-key", default=None, type=str,
                              help="server secret key file")
    group_config.add_argument("-ck", "--cli-secret-key", default=None, type=str,
                              help="client secret key file")
    args = parser.parse_args()
    return args


def _init_config_file():
    vega_dir = os.path.join(os.getenv("HOME"), ".vega")
    os.makedirs(vega_dir, exist_ok=True)
    vega_config_file = os.path.join(vega_dir, "vega.ini")
    if os.path.exists(vega_config_file):
        print("vega config file ({}) already exists.".format(vega_config_file))
        return
    with open(vega_config_file, "w") as f:
        f.write("[security]\n")
        f.write("enable=True\n")
        f.write("\n")
        f.write("[https]\n")
        f.write("cert_pem_file=\n")
        f.write("secret_key_file=\n")
        f.write("\n")
        f.write("[limit]\n")
        f.write("request_frequency_limit=100/minute\n")
        f.write("max_content_length=1000000000\n")
        f.write("#white_list=0.0.0.0,127.0.0.1\n")
    print("initializing vega config file ({}).".format(vega_config_file))


def _process_cmd(args):
    if args.init:
        _init_config_file()
        return
    config = read_config_file()
    if not config:
        return
    if args.query:
        config = _process_cmd_query(args, config)
        return
    if args.set is not None:
        if args.set == 1:
            config.set("security", "enable", "True")
            print("set vega security mode to True")
        else:
            config.set("security", "enable", "False")
            print("set vega security mode to False")
    elif args.module is not None:
        config = _process_cmd_module(args, config)
    with open(os.path.join(os.environ['HOME'], ".vega", "vega.ini"), "w") as f:
        config.write(f)


def _process_cmd_query(args, config):
    if args.query == "sec":
        print(str(config["security"]["enable"]))
    elif args.query == "https":
        print("cert_pem_file: {}".format(
            config["https"]["cert_pem_file"] if "cert_pem_file" in config["https"] else None))
        print("secret_key_file: {}".format(
            config["https"]["secret_key_file"] if "secret_key_file" in config["https"] else None))
    return config


def _process_cmd_module(args, config):
    print("set cert/key file of module {}".format(args.module))
    if args.module == "https":
        if args.cert:
            config.set("https", "cert_pem_file", args.cert)
        if args.secret_key:
            config.set("https", "secret_key_file", args.secret_key)
    return config


def vega_config_operate():
    """Run pipeline."""
    args = _parse_args()
    _process_cmd(args)


if __name__ == '__main__':
    vega_config_operate()
