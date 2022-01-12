# -*- coding: utf-8 -*-

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
"""Manage Loss class."""
import logging
from inspect import isclass
from functools import partial
import vega
from vega.common import ClassFactory, ClassType
from vega.trainer.modules.config_bakcend_map import ConfigBackendMapping
from vega.trainer.modules.conf.loss import LossConfig, LossMappingDict


class Loss(object):
    """Register and call loss class."""

    config = LossConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch loss as default
        raw_config = self.config.to_dict()
        raw_config.type = self.config.type
        map_dict = LossMappingDict()
        self.map_config = ConfigBackendMapping(
            map_dict.type_mapping_dict, map_dict.params_mapping_dict).backend_mapping(raw_config)
        self._cls = ClassFactory.get_cls(ClassType.LOSS, self.map_config.type)

    def __call__(self):
        """Call loss cls."""
        params = self.map_config.get("params", {})
        logging.debug("Call Loss. name={}, params={}".format(self._cls.__name__, params))
        try:
            if params:
                cls_obj = self._cls(**params) if isclass(self._cls) else partial(self._cls, **params)
            else:
                cls_obj = self._cls() if isclass(self._cls) else partial(self._cls)
            if vega.is_torch_backend():
                if vega.is_gpu_device():
                    cls_obj = cls_obj.cuda()
                elif vega.is_npu_device() and not cls_obj.__class__.__name__ == 'SumLoss':
                    cls_obj = cls_obj.to(vega.get_devices())
            return cls_obj
        except Exception as ex:
            logging.error("Failed to call Loss name={}, params={}".format(self._cls.__name__, params))
            raise ex


if vega.is_torch_backend():
    import torch.nn as torch_nn

    ClassFactory.register_from_package(torch_nn, ClassType.LOSS)
    try:
        import timm.loss as timm_loss

        ClassFactory.register_from_package(timm_loss, ClassType.LOSS)
    except Exception:
        logging.debug('Falied to get timm loss.')
elif vega.is_tf_backend():
    import tensorflow.compat.v1.losses as tf_loss

    ClassFactory.register_from_package(tf_loss, ClassType.LOSS)
elif vega.is_ms_backend():
    import mindspore.nn.loss as ms_loss

    ClassFactory.register_from_package(ms_loss, ClassType.LOSS)
