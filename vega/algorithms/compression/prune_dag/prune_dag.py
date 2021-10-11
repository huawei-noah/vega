# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is Operator SearchSpace."""
import copy
import logging
import os
import vega
from vega.common import ClassFactory, ClassType
from vega.common.general import TaskConfig
from vega.trainer.callbacks import Callback
from vega.core.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig
from .prune_events import prune_dag_model
from .dag_relations import node_relations_search, is_conv2d
from vega.model_zoo import ModelZoo
from vega.common.parameter_sharing import ParameterSharing


@ClassFactory.register(ClassType.SEARCHSPACE)
class PruneDAGSearchSpace(SearchSpace):
    """Prune SearchSpace."""

    @classmethod
    def get_space(self, desc):
        """Get model and input."""
        self.model = ModelZoo().get_model(PipeStepConfig.model.model_desc, PipeStepConfig.model.pretrained_model_file)
        arch_params_key = '{}.out_channels'
        search_space = [dict(key=arch_params_key.format(name), type="HALF", range=[module.out_channels])
                        for name, module in self.model.named_modules() if is_conv2d(module)]
        return {"hyperparameters": search_space}

    @classmethod
    def to_desc(self, desc):
        """Decode to model desc."""
        pruned_model = copy.deepcopy(self.model)
        node_relations_search(pruned_model, desc)
        prune_dag_model(pruned_model)
        PipeStepConfig.model.pretrained_model_file = ParameterSharing().push(pruned_model, 'pruned_weights')
        return pruned_model.to_desc()


@ClassFactory.register(ClassType.SEARCHSPACE)
class SCOPDAGSearchSpace(SearchSpace):
    """SCOP DAG SearchSpace."""

    @classmethod
    def get_space(self, desc):
        """Get model and input."""
        self.model = ModelZoo().get_model(PipeStepConfig.model.model_desc, PipeStepConfig.model.pretrained_model_file)
        if not desc.get("hyperparameters"):
            raise ValueError("hyperparameters should be config in SCOPDAGSearchSpace.")
        search_space = []
        for item in desc.get("hyperparameters"):
            arch_params_key = "{}." + item.get("key")
            arch_type = item.get("type")
            arch_type_range = item.get("range")
            search_space.extend([dict(key=arch_params_key.format(name), type=arch_type, range=arch_type_range)
                                 for name, module in self.model.named_modules() if is_conv2d(module)])
        # first conv not pruned.
        search_space.pop(0)
        return {"hyperparameters": search_space}

    @classmethod
    def to_desc(self, desc):
        """Decode to model desc."""
        pruned_model = copy.deepcopy(self.model)
        desc = self._decode_fn(pruned_model, desc)
        node_relations_search(pruned_model, desc)
        prune_dag_model(pruned_model)
        PipeStepConfig.model.pretrained_model_file = ParameterSharing().push(pruned_model, 'pruned_weights')
        return pruned_model.to_desc()

    @classmethod
    def _decode_fn(self, model, desc):
        mask_code_desc = {}
        kf_scale_dict = self._load_kf_scale()
        if kf_scale_dict:
            logging.info("Start prune with kf scale.")
        for name, rate in desc.items():
            node_name = '.'.join(name.split('.')[:-1])
            arch_type = name.split('.')[-1]
            if node_name not in model.module_map:
                continue
            node_channels = model.module_map[node_name].module.out_channels
            if arch_type == 'prune_d_rate':
                select_idx = round(node_channels * rate / 100 / 16) * 16
                select_idx = select_idx if select_idx > 16 else node_channels
            else:
                select_idx = node_channels * rate // 100
            if kf_scale_dict:
                beta = kf_scale_dict.get(node_name + ".kf_scale").cpu()
                next_node = model.module_map[node_name].child_nodes[0]
                bn_weight = 1
                if next_node.module_type == "BatchNorm2d":
                    bn_weight = next_node.module.weight.data.abs().cpu()
                score = bn_weight * (beta - (1 - beta)).squeeze()
                _, idx = score.sort()
                pruned_idx = idx[select_idx:].numpy().tolist()
                idx_code = [1 if idx in pruned_idx else 0 for idx in range(node_channels)]
            else:
                idx_code = [1 if idx < select_idx else 0 for idx in range(node_channels)]
            mask_code_desc[node_name + '.out_channels'] = idx_code
        return mask_code_desc

    @classmethod
    def _load_kf_scale(cls):
        if not PipeStepConfig.model.kf_sacle_file:
            return
        import torch
        file_path = PipeStepConfig.model.kf_sacle_file
        file_path = file_path.replace("{local_base_path}", os.path.join(TaskConfig.local_base_path, TaskConfig.task_id))
        return torch.load(file_path)


@ClassFactory.register(ClassType.CALLBACK)
class AdaptiveBatchNormalizationCallback(Callback):
    """Adaptive Batch Normalization."""

    def before_train(self, logs=None):
        """Freeze Conv2D and BatchNorm."""
        if not vega.is_torch_backend():
            return
        import torch
        for name, module in self.trainer.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                for name, parameter in module.named_parameters():
                    parameter.requires_grad_(False)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
        learnable_params = [param for param in self.trainer.model.parameters() if param.requires_grad]
        logging.info("Adaptive BatchNormalization learnable params size: {}".format(len(learnable_params)))
