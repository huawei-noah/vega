# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""
from .modules.conf.loss import LossConfig
from .modules.conf.lr_scheduler import LrSchedulerConfig
from .modules.conf.optim import OptimConfig
from zeus.common import ConfigSerializable


class MetricsConfig(ConfigSerializable):
    """Default Metrics Config."""

    _class_type = "trainer.metric"
    _update_all_attrs = True
    type = 'accuracy'
    params = {}

    @classmethod
    def from_json(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(MetricsConfig, cls).from_json(data, skip_check)
        if "params" not in data:
            cls.params = {}
        return cls

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        check_rules = {"type": {"type": str},
                       "params": {"type": dict}}
        return check_rules


class TrainerConfig(ConfigSerializable):
    """Default Trainer Config."""

    type = 'Trainer'
    with_valid = True
    cuda = True
    is_detection_trainer = False
    distributed = False
    save_model_desc = False
    report_freq = 10
    seed = 0
    epochs = 1
    valid_interval = 1
    syncbn = False
    amp = False
    lazy_built = False
    callbacks = None
    grad_clip = None
    pretrained_model_file = None
    model_statistics = True
    device = cuda if cuda is not True else 0
    # config a object
    optimizer = OptimConfig
    lr_scheduler = LrSchedulerConfig
    metric = MetricsConfig
    loss = LossConfig
    # TODO: need to delete
    limits = None
    init_model_file = None
    pareto_front_file = None
    unrolled = True
    save_best_model = False
    model_arch = None
    model_desc_file = None
    codec = None
    model_desc = None
    hps_file = None
    hps_folder = None
    save_model_descs = None
    random_file = None
    loss_scale = 1.
    save_steps = 500
    report_on_valid = False
    perfs_cmp_mode = None
    perfs_cmp_key = None
    call_metrics_on_train = True
    lr_adjustment_position = 'after_epoch'
    report_on_epoch = False
    calc_params_each_epoch = False
    model_path = None
    checkpoint_file = None
    weights_file = None
    get_train_metric_after_epoch = True
    kwargs = None
    train_verbose = 2
    valid_verbose = 2
    train_report_steps = 10
    valid_report_steps = 10
    load_checkpoint = True

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        check_rules_trainer = {"type": {"type": str},
                               "epochs": {"type": int},
                               "optimizer": {"type": dict},
                               "lr_scheduler": {"type": dict},
                               "loss": {"type": dict},
                               "metric": {"type": dict},
                               "calc_params_each_epoch": {"type": bool}
                               }
        return check_rules_trainer
