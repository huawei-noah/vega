# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""
import argparse
import os


_pkgs = [
    "scikit-build==0.11.1",
    "opencv-python-headless==4.3.0.38",
    "albumentations==0.4.6",
    "torchnet==0.0.4",
    "Cython==0.29.21",
    "addict==2.2.1",
    "distributed==2.18.0",
    "enum34==1.1.6",
    "pandas==0.25.2",
    "scikit-learn==0.21.3",
    "tensorboardX==1.9",
    "terminaltables==3.1.0",
    "timm==0.1.18",
    "pareto==1.1.1.post3",
    "py-dag==3.0.1",
    "thop==0.0.31.post2004101309",
    "torchvision==0.4.1",
    "pytest-runner==5.2",
    "mmcv==0.2.14",
    "cffi==1.14.1",
    "horovod==0.19.3",
    "tensorflow-gpu>=1.14.0,<2.0",
    "intervaltree==3.0.2",
    "more-itertools==8.2.0",
    "imagecorruptions==1.0.0",
    "ujson==3.0.0",
    "PrettyTable==0.7.2",
    "imgaug==0.4.0",
    "pycocotools==2.0.1",
    "matplotlib==3.3.0",
    "tf-models-official==0.0.3.dev1",
    "mindspore==0.7.0",
    "PyYAML==5.1.2",
]


def _gen_cmd_file(args):
    pip = args.pip_version
    output_file = args.output_script_name
    output = ["{} install --upgrade pip".format(pip)]
    template = "{} install --user \"{}\""
    horovod = "HOROVOD_GPU_OPERATIONS=NCCL {} install --user --no-cache-dir \"{}\""
    for name_version in _pkgs:
        if "horovod" in name_version:
            cmd = horovod.format(pip, name_version)
        else:
            cmd = template.format(pip, name_version)
        output.append(cmd)
    output = "\n".join(output)
    try:
        with open(output_file, "w") as f:
            f.write(output)
        print("The installation script has been generated: {}".format(output_file))
        return output_file
    except Exception as e:
        print("Failed to generate installation script: {}".format(output_file))
        raise e


def _exec_shell(_file):
    if os.path.exists(_file):
        os.system("bash {}".format(_file))


def _parse_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument("-b", "--backend", default="pytorch", type=str, help="pytorch|tensorflow|mindspore")
    parser.add_argument("-o", "--output_script_name", default="./install_dependencies.sh", type=str,
                        help="installation script name, the default value is ./install_dependencies.sh")
    parser.add_argument("-e", "--exec_script", default="t", type=str,
                        choices=["true", "t", "false", "f", "yes", "y", "no", "n"],
                        help="whether to execute the installation script, the defualt value is true")
    parser.add_argument("-p", "--pip_version", default="pip3", type=str, choices=["pip3", "pip"],
                        help="pip version, the default value is pip3")
    args = parser.parse_args()
    return args


def _install_pkgs():
    args = _parse_args("Install vega dependent packages.")
    _file = _gen_cmd_file(args)
    if args.exec_script in ["true", "t", "yes", "y"]:
        _exec_shell(_file)


if __name__ == "__main__":
    _install_pkgs()
