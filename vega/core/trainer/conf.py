# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""


class OptimConfig(object):
    """Default Optim Config."""

    _class_type = "trainer.optim"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'Adam'
    params = {"lr": 0.1}


class LrSchedulerConfig(object):
    """Default LrScheduler Config."""

    _class_type = "trainer.lr_scheduler"
    _update_all_attrs = True
    _exclude_keys = ['type']
    type = 'MultiStepLR'
    params = {"milestones": [75, 150], "gamma": 0.5}


class MetricsConfig(object):
    """Default Metrics Config."""

    _class_type = "trainer.metric"
    _update_all_attrs = True
    type = 'accuracy'
    params = {}


class LossConfig(object):
    """Default Loss Config."""

    _class_type = "trainer.loss"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'CrossEntropyLoss'
    params = {}


class TrainerConfig(object):
    """Default Trainer Config."""

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
    report_verbose = 2
    device = cuda if cuda is not True else 0
    # config a object
    optim = OptimConfig
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
