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
"""Default task config."""


DEFAULT_CONFIG = {
    "Classification": {
        "optimizer": {"type": "Adam", "params": {"lr": 0.1}},
        "lr_scheduler": {"type": "MultiStepLR", "params": {"milestones": [75, 150], "gamma": 0.5}},
        "loss": {"type": "CrossEntropyLoss"},
        "metric": {"type": "accuracy"},
    },
    "Detection": {
        "optimizer": {"type": "SGD", "params": {"lr": 0.003, "momentum": 0.9, "weight_decay": 0.0001}},
        "lr_scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": 30000, "eta_min": 0.0001}},
        "loss": {"type": "SumLoss"},
        "metric": {"type": "coco",
                   "params": {"anno_path": "/cache/datasets/COCO2017/annotations/instances_val2017.json"}},
    },
    "Segmentation": {
        "optimizer": {"type": "Adam", "params": {"lr": 5e-5}},
        "lr_scheduler": {"type": "StepLR", "params": {"step_size": 5, "gamma": 0.5}},
        "loss": {"type": "CrossEntropyLoss", "params": {"ignore_index": 255}},
        "metric": {"type": "IoUMetric", "params": {"num_class": 21}},
    },
    "SuperResolution": {
        "optimizer": {"type": "Adam", "params": {"lr": 0.0004}},
        "lr_scheduler": {"type": "MultiStepLR", "params": {"milestones": [100, 200], "gamma": 0.5}},
        "loss": {"type": "L1Loss"},
        "metric": {"type": "PSNR", "params": {"scale": 2}},
    },
}
