# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch data providers."""
from modnas.utils import merge_config
from modnas.registry.data_provider import build
from modnas.registry.dataloader import build as build_dataloader
from modnas.registry.dataset import build as build_dataset


def get_data(configs):
    """Return a new dataset."""
    config = None
    for conf in configs:
        if conf is None:
            continue
        config = conf if config is None else merge_config(config, conf)
    if config is None:
        return None
    return build_dataset(config)


def get_data_provider(config):
    """Return a new DataProvider."""
    trn_data = get_data([config.get('data'), config.get('train_data')])
    val_data = get_data([config.get('data'), config.get('valid_data')])
    dloader_conf = config.get('data_loader', None)
    data_prov_conf = config.get('data_provider', {})
    data_provd_args = data_prov_conf.get('args', {})
    if dloader_conf is not None:
        trn_loader, val_loader = build_dataloader(dloader_conf,
                                                  trn_data=trn_data,
                                                  val_data=val_data)
        data_provd_args['train_loader'] = trn_loader
        data_provd_args['valid_loader'] = val_loader
    elif not data_prov_conf:
        return None
    data_prov = build(data_prov_conf.get('type', 'DefaultDataProvider'), **data_provd_args)
    return data_prov
