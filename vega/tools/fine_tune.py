# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fine tune vega model."""

import pandas as pd
import logging
import json
from vega.common import argment_parser
from vega.common.general import General
from vega.common.task_ops import TaskOps
from vega.common.file_ops import FileOps
from vega.core.pipeline.conf import PipelineConfig, PipeStepConfig
from vega.trainer.conf import TrainerConfig
from vega.core.pipeline.train_pipe_step import TrainPipeStep
from vega.tools.init_env import _init_env
from vega.tools.run_pipeline import _set_backend


def _parse_args():
    parser = argment_parser("Fine tune DNet model or ResNet model.")
    group_backend = parser.add_argument_group(title="Set backend and device, default is pytorch and GPU")
    group_backend.add_argument("-b", "--backend", default="pytorch", type=str,
                               choices=["pytorch", "p", "tensorflow", "t", "mindspore", "m"],
                               help="set training platform")
    group_backend.add_argument("-d", "--device", default="GPU", type=str,
                               choices=["GPU", "NPU"],
                               help="set training device")
    group_dataset = parser.add_argument_group(title="Dataset setting")
    group_dataset.add_argument("-ds", "--dataset", default=None, type=str, required=True,
                               help="dataset type, eg. Cifar10, ClassificationDataset.")
    group_dataset.add_argument("-dp", "--data_path", default=None, type=str, required=True,
                               help="dataset path.")
    group_dataset.add_argument("-bs", "--batch_size", default=None, type=int, required=True,
                               help="dataset batch size.")
    group_dataset.add_argument("-tp", "--train_portion", default=1.0, type=float,
                               help="train portion.")
    group_dataset.add_argument("-is", "--image_size", default=224, type=int,
                               help="image size.")
    group_trainer = parser.add_argument_group(title="Trainer setting")
    group_trainer.add_argument("-e", "--epochs", default=40, type=int,
                               help="Modify fully_train epochs")
    group_model = parser.add_argument_group(title="model setting")
    group_model.add_argument("-n", "--network", default=None, type=str,
                             choices=["dnet", "resnet"],
                             help="network name, dnet or resnet.")
    # denet
    group_model.add_argument("-de", "--dnet_encoding", default=None, type=str,
                             help="DNet network Encoding")
    # resnet
    group_model.add_argument("-rd", "--resnet_depth", default=50, type=int,
                             help="ResNet network depth")
    # general
    group_model.add_argument("-mf", "--pretrained_model_file", default=None, type=str, required=True,
                             help="pretrained model file")
    group_model.add_argument("-nc", "--num_classes", default=None, type=int, required=True,
                             help="number of classes")
    group_output = parser.add_argument_group(title="output setting")
    group_output.add_argument("-o", "--output_path", default=None, type=int,
                              help="set output path")
    args = parser.parse_args()
    return args


def _set_pipeline_config(args):
    General.step_name = "fine_tune"
    PipelineConfig.steps = ["fine_tune"]
    PipeStepConfig.type = "TrainPipeStep"


def _set_dataset_config(args):
    PipeStepConfig.dataset.from_dict({
        "type": args.dataset,
        "common": {
            "data_path": args.data_path,
            "batch_size": args.batch_size,
            "train_portion": args.train_portion,
        },
        "train": {
            "transforms": [
                {"type": "Resize", "size": [args.image_size + 32, args.image_size + 32]},
                {"type": "RandomCrop", "size": [args.image_size, args.image_size]},
                {"type": "RandomHorizontalFlip"},
                {"type": "ToTensor"},
                {"type": "Normalize", "mean": [0.50, 0.5, 0.5], "std": [0.50, 0.5, 0.5]},
            ]
        },
        "val": {
            "transforms": [
                {"type": "Resize", "size": [args.image_size, args.image_size]},
                {"type": "ToTensor"},
                {"type": "Normalize", "mean": [0.50, 0.5, 0.5], "std": [0.50, 0.5, 0.5]},
            ]
        },
        "test": {
            "transforms": [
                {"type": "Resize", "size": [args.image_size, args.image_size]},
                {"type": "ToTensor"},
                {"type": "Normalize", "mean": [0.50, 0.5, 0.5], "std": [0.50, 0.5, 0.5]},
            ]
        },
    })


def _set_model_config(args):
    if args.network == "dnet":
        config = {
            "model_desc": {
                "type": "DNet",
                "n_class": args.num_classes,
                "encoding": args.dnet_encoding,
            },
            "pretrained_model_file": args.pretrained_model_file,
            "head": "fc",
        }
        if args.backend in ["mindspore", "m"]:
            config = {
                "model_desc": {
                    "modules": ["backbone"],
                    "backbone": {
                        "type": "DNet",
                        "n_class": args.num_classes,
                        "encoding": args.dnet_encoding,
                    },
                },
                "pretrained_model_file": args.pretrained_model_file,
            }
    elif args.network == "resnet":
        config = {
            "model_desc": {
                "type": "ResNetTF",
                "resnet_size": args.resnet_depth,
                "num_classes": args.num_classes,
            },
            "pretrained_model_file": args.pretrained_model_file,
            "head": "resnet_model/dense/",
        }
    else:
        raise Exception("Not supported network: {}".format(args.network))
    PipeStepConfig.model.from_dict(config)


def _set_trainer_config(args):
    config = {
        "epochs": args.epochs,
        "loss": {
            "type": "CrossEntropyLoss",
            "params": {"sparse": True},
        },
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.003,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        },
        "lr_scheduler": {
            "type": "WarmupScheduler",
            "by_epoch": False,
            "params": {
                "warmup_type": "linear",
                "warmup_iters": 500,
                "warmup_ratio": 0.01,
                "after_scheduler_config": {
                    "type": "MultiStepLR",
                    "by_epoch": True,
                    "params": {
                        "milestones": [30],
                        "gamma": 0.1,
                    },
                },
            },
        },
    }
    if args.backend in ["pytorch", "p"]:
        pass
    elif args.backend in ["tensorflow", "t"]:
        config["lr_scheduler"]["by_epoch"] = True
        config["lr_scheduler"]["params"]["warmup_iters"] = 5
    elif args.backend in ["mindspore", "m"]:
        config["lr_scheduler"]["params"]["warmup_ratio"] = 0.00001
        config["optimizer"] = {
            "type": "Adam",
            "params": {"lr": 0.0001},
        }
    else:
        raise Exception("Not")
    TrainerConfig.from_dict(config)


def _show_performance():
    output_file = FileOps.join_path(
        TaskOps().local_output_path, General.step_name, "output.csv")
    try:
        data = pd.read_csv(output_file)
    except Exception:
        logging.info("  Result file output.csv is not existed or empty.")
        return
    if data.shape[1] < 2 or data.shape[0] == 0:
        logging.info("  Result file output.csv is empty.")
        return
    logging.info("-" * 48)
    data = json.loads(data.to_json())
    logging.info("  result: {}".format(data["performance"]["0"]))
    logging.info("-" * 48)


def _fine_tune():
    args = _parse_args()
    _set_backend(args)
    _set_pipeline_config(args)
    _set_dataset_config(args)
    _set_model_config(args)
    _set_trainer_config(args)
    _init_env()
    TrainPipeStep().do()
    _show_performance()


if __name__ == "__main__":
    _fine_tune()
