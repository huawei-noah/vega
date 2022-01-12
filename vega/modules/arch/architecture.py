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
"""Architecture."""
import os
import logging
from vega.common.class_factory import ClassFactory
from vega.modules.arch.combiner import ConnectionsArchParamsCombiner
from vega.networks.network_desc import NetworkDesc
import vega
from vega.core.pipeline.conf import PipeStepConfig


def change_model_rebuild(model, changed_name_list, mask_weight_list, pretrained_model_file):
    """Change model and rebuild."""
    from mindspore.train.serialization import load_checkpoint
    if len(changed_name_list) == len(mask_weight_list):
        model_desc = model.desc
        for changed_name, mask in zip(changed_name_list, mask_weight_list):
            name = changed_name.split('.')
            if len(name) <= 6:
                if len(name) == 6:
                    model_desc[name[0]][name[1]][name[2]][name[3]][name[4]][name[5]] = sum(mask)
                if len(name) == 5:
                    model_desc[name[0]][name[1]][name[2]][name[3]][name[4]] = sum(mask)
                if len(name) == 4:
                    model_desc[name[0]][name[1]][name[2]][name[3]] = sum(mask)
                if len(name) == 3:
                    model_desc[name[0]][name[1]][name[2]] = sum(mask)
                if len(name) == 2:
                    model_desc[name[0]][name[1]] = sum(mask)
            else:
                raise ValueError('Name must be shorter than 6.')
        network = NetworkDesc(model_desc)
        model = network.to_model()
        model_desc.pop('_arch_params') if '_arch_params' in model_desc else model_desc
        model.desc = model_desc
        if pretrained_model_file and hasattr(model, "pretrained"):
            pretrained_weight = model.pretrained(pretrained_model_file)
            load_checkpoint(pretrained_weight, net=model)
            os.remove(pretrained_weight)
        return model
    else:
        raise ValueError('Name and weight do not match.')


def transform_architecture(model, pretrained_model_file=None):
    """Transform architecture."""
    if not hasattr(model, "_arch_params") or not model._arch_params or \
            PipeStepConfig.pipe_step.get("type") == "TrainPipeStep":
        return model
    model._apply_names()
    logging.info("Start to transform architecture, model arch params type: {}".format(model._arch_params_type))
    ConnectionsArchParamsCombiner().combine(model)
    if vega.is_ms_backend():
        from mindspore.train.serialization import load_checkpoint
        changed_name_list = []
        mask_weight_list = []
        for name, module in model.named_modules():
            if not ClassFactory.is_exists(model._arch_params_type, module.model_name):
                continue
            changed_name_list, mask_weight_list = decode_fn_ms(module, changed_name_list, mask_weight_list)
        model = change_model_rebuild(model, changed_name_list, mask_weight_list, pretrained_model_file)
    else:
        for name, module in model.named_modules():
            if not ClassFactory.is_exists(model._arch_params_type, module.model_name):
                continue
            arch_cls = ClassFactory.get_cls(model._arch_params_type, module.model_name)
            decode_fn(module, arch_cls)
            module.register_forward_pre_hook(arch_cls.fit_weights)
    return model


def register_clear_module_arch_params_hooks(model):
    """Register hooks."""
    if not hasattr(model, "_arch_params") or not model._arch_params or \
            PipeStepConfig.pipe_step.get("type") == "TrainPipeStep":
        return
    for name, module in model.named_modules():
        if not ClassFactory.is_exists(model._arch_params_type, module.model_name):
            continue
        module.register_forward_hook(module.clear_module_arch_params)


def decode_fn(module, arch_cls):
    """Decode function."""
    for name, value in module._arch_params.items():
        module_name = '.'.join(name.split('.')[:-1])
        if module.name != module_name:
            continue
        module_attr = name.split('.')[-1]
        org_value = getattr(module, module_attr)
        setattr(module, module_attr, arch_cls.decode(value, org_value))


def decode_fn_ms(module, changed_name_list, mask_weight_list):
    """Decode function."""
    for name, value in module._arch_params.items():
        module_name = '.'.join(name.split('.')[:-1])
        if module.name == module_name:
            module_attr = name.split('.')[-1]
            changed_name = module.name.split("/")[0] + '.' + module_attr
            changed_name_list.append(changed_name)
            mask_weight_list.append(value)
    return changed_name_list, mask_weight_list


class Architecture(object):
    """Architecture base class."""

    @staticmethod
    def fit_weights(module, x):
        """Fit weights."""
        raise NotImplementedError

    @staticmethod
    def decode(value, org_value):
        """Decode function."""
        raise NotImplementedError
