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

"""Torch data providers."""
import copy
from modnas.utils.config import merge_config
from modnas.registry.data_provider import build
from modnas.registry.dataloader import build as build_dataloader
from modnas.registry.dataset import build as build_dataset


def get_data(configs):
    """Return a new dataset."""
    config = None
    for conf in configs:
        if conf is None:
            continue
        config = copy.deepcopy(conf) if config is None else merge_config(config, conf)
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
