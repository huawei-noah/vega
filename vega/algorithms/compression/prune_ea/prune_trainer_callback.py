# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer for searching pruned model."""
import logging
import os
import numpy as np
import copy
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.utils.data
from vega.core.trainer.pytorch import Trainer
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import Config
from vega.core.metrics.pytorch import calc_model_flops_params
from vega.search_space import SearchSpace
from vega.search_space.networks import NetworkDesc
from vega.datasets.pytorch import Dataset
from .prune_codec import PruneCodec
from vega.core.common.file_ops import FileOps
from vega.core.visual import dump_trainer_visual_info
from vega.core.common.utils import update_dict
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class PruneTrainerCallback(Callback):
    """Callback of Prune Trainer."""

    def before_train(self, logs=None):
        """Be called before the train process."""
        self.cfg = self.trainer.cfg
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        self.device = self.cfg.device
        self.base_net_desc = SearchSpace().cfg
        self.model = self.trainer.model
        if self.model is None:
            self.model = self.trainer._init_model()
        self.model._init_weights()
        self.model = self.model.to(self.device)
        count_input = torch.FloatTensor(1, 3, 32, 32).to(self.device)
        self.flops_count, self.params_count = calc_model_flops_params(
            self.model, count_input)
        if not self.validate():
            return
        self.model = self._generate_init_model(self.model).to(self.device)
        self.trainer.build(self.model)

    def after_epoch(self, epoch, logs=None):
        """Be called after one epoch training."""
        self.summary_perfs = logs.get('summary_perfs', None)
        if self.summary_perfs['best_valid_perfs_changed']:
            self._save_best_model()

    def after_train(self, logs=None):
        """Be called after the whole train process."""
        self.metric = list(self.summary_perfs['best_valid_perfs'].values())[0][0]
        self.save_metrics_value()
        if self.cfg.get('save_model_desc', False):
            self._save_model_desc()

    def _load_model(self, model_prune):
        """Load model from file.

        :param model_prune: searched pruned model
        :type model_prune: torch.nn.Module
        :return: initial model after loading pretrained model
        :rtype: torch.nn.Module
        """
        init_model_file = self.cfg.init_model_file
        if ":" in init_model_file:
            local_path = FileOps.join_path(
                self.trainer.get_local_worker_path(), os.path.basename(init_model_file))
            FileOps.copy_file(init_model_file, local_path)
            init_model_file = local_path
        checkpoint = torch.load(init_model_file)
        network_desc = copy.deepcopy(self.base_net_desc)
        network_desc.backbone.chn = network_desc.backbone.base_chn
        network_desc.backbone.chn_node = network_desc.backbone.base_chn_node
        network_desc.backbone.encoding = model_prune.encoding
        model_init = NetworkDesc(network_desc).to_model()
        model_init.load_state_dict(checkpoint)
        return model_init

    def _init_chn_node_mask(self, model_prune):
        """Init channel node mask.

        :param model_prune: searched pruned model
        :type model_prune: torch.nn.Module
        :return: channel node masks
        :rtype: array
        """
        if model_prune.chn_mask is None:
            Codec = PruneCodec('PruneCodec', SearchSpace())
            net_desc = Codec.decode(model_prune.encoding)
            model_prune.chn_mask = net_desc._desc.backbone.chn_mask
            model_prune.chn_node_mask = net_desc._desc.backbone.chn_node_mask
        chn_node_mask_tmp = model_prune.chn_node_mask
        chn_node_mask = []
        for i, single_mask in zip([1, 3, 3, 3], chn_node_mask_tmp):
            for _ in range(i):
                chn_node_mask.append(single_mask)
        return chn_node_mask

    def _generate_init_model(self, model_prune):
        """Generate init model by loading pretrained model.

        :param model_prune: searched pruned model
        :type model_prune: torch.nn.Module
        :return: initial model after loading pretrained model
        :rtype: torch.nn.Module
        """
        model_init = self._load_model(model_prune)
        chn_node_mask = self._init_chn_node_mask(model_prune)
        chn_node_id = 0
        chn_id = 0
        chn_mask = model_prune.chn_mask
        start_mask = []
        end_mask = []
        for name, m1 in model_init.named_modules():
            if name == 'conv1':
                end_mask = chn_node_mask[chn_node_id]
                end_mask = np.asarray(end_mask)
                idx1 = np.squeeze(np.argwhere(np.asarray(
                    np.ones(end_mask.shape) - end_mask)))
                mask = np.ones(m1.weight.data.shape)
                mask[idx1.tolist(), :, :, :] = 0
                m1.weight.data = m1.weight.data * torch.FloatTensor(mask)
                m1.weight.data[idx1.tolist(), :, :, :].requires_grad = False
                chn_node_id += 1
                continue
            if name == 'bn1':
                idx1 = np.squeeze(np.argwhere(np.asarray(
                    np.ones(end_mask.shape) - end_mask)))
                mask = np.ones(m1.weight.data.shape)
                mask[idx1.tolist()] = 0
                m1.weight.data = m1.weight.data * torch.FloatTensor(mask)
                m1.bias.data = m1.bias.data * torch.FloatTensor(mask)
                m1.running_mean = m1.running_mean * torch.FloatTensor(mask)
                m1.running_var = m1.running_var * torch.FloatTensor(mask)
                m1.weight.data[idx1.tolist()].requires_grad = False
                m1.bias.data[idx1.tolist()].requires_grad = False
                m1.running_mean[idx1.tolist()].requires_grad = False
                m1.running_var[idx1.tolist()].requires_grad = False
                continue
            if isinstance(m1, model_init.block):
                conv_id = 0
                for layer1 in m1.modules():
                    if isinstance(layer1, nn.Conv2d):
                        if conv_id == 0:
                            start_mask = chn_node_mask[chn_node_id - 1]
                            end_mask = chn_mask[chn_id]
                            chn_id += 1
                        if conv_id == 1:
                            start_mask = end_mask
                            end_mask = chn_node_mask[chn_node_id]
                        # shortcut
                        if conv_id == 2:
                            start_mask = chn_node_mask[chn_node_id - 1]
                            end_mask = chn_node_mask[chn_node_id]
                        start_mask = np.asarray(start_mask)
                        end_mask = np.asarray(end_mask)
                        idx0 = np.squeeze(np.argwhere(
                            np.asarray(np.ones(start_mask.shape) - start_mask)))
                        idx1 = np.squeeze(np.argwhere(
                            np.asarray(np.ones(end_mask.shape) - end_mask)))
                        mask = np.ones(layer1.weight.data.shape)
                        mask[:, idx0.tolist(), :, :] = 0
                        mask[idx1.tolist(), :, :, :] = 0
                        layer1.weight.data = layer1.weight.data * \
                            torch.FloatTensor(mask)
                        layer1.weight.data[:, idx0.tolist(
                        ), :, :].requires_grad = False
                        layer1.weight.data[idx1.tolist(
                        ), :, :, :].requires_grad = False
                        conv_id += 1
                        continue
                    if isinstance(layer1, nn.BatchNorm2d):
                        idx1 = np.squeeze(np.argwhere(
                            np.asarray(np.ones(end_mask.shape) - end_mask)))
                        mask = np.ones(layer1.weight.data.shape)
                        mask[idx1.tolist()] = 0
                        layer1.weight.data = layer1.weight.data * \
                            torch.FloatTensor(mask)
                        layer1.bias.data = layer1.bias.data * \
                            torch.FloatTensor(mask)
                        layer1.running_mean = layer1.running_mean * \
                            torch.FloatTensor(mask)
                        layer1.running_var = layer1.running_var * \
                            torch.FloatTensor(mask)
                        layer1.weight.data[idx1.tolist()].requires_grad = False
                        layer1.bias.data[idx1.tolist()].requires_grad = False
                        layer1.running_mean[idx1.tolist()].requires_grad = False
                        layer1.running_var[idx1.tolist()].requires_grad = False
                # every BasicBlock +1
                chn_node_id += 1
            if isinstance(m1, nn.Linear):
                idx1 = np.squeeze(np.argwhere(
                    np.asarray(np.ones(end_mask.shape) - end_mask)))
                mask = np.ones(m1.weight.data.shape)
                mask[:, idx1.tolist()] = 0
                m1.weight.data = m1.weight.data * torch.FloatTensor(mask)
                m1.weight.data[:, idx1.tolist()].requires_grad = False
        return model_init

    def _save_model_desc(self):
        """Save final model desc of NAS."""
        pf_file = FileOps.join_path(self.trainer.local_output_path, self.trainer.step_name, "pareto_front.csv")
        if not FileOps.exists(pf_file):
            return
        with open(pf_file, "r") as file:
            pf = pd.read_csv(file)
        pareto_fronts = pf["encoding"].tolist()
        search_space = SearchSpace()
        codec = PruneCodec('PruneCodec', search_space)
        for i, pareto_front in enumerate(pareto_fronts):
            pareto_front = [int(x) for x in pareto_front[1:-1].split(',')]
            model_desc = Config()
            model_desc.modules = search_space.search_space.modules
            model_desc.backbone = codec.decode(pareto_front)._desc.backbone
            self.trainer.output_model_desc(i, model_desc)

    def save_metrics_value(self):
        """Save the metric value of the trained model.

        :return: save_path (local) and s3_path (remote). If s3_path not specified, then s3_path is None
        :rtype: a tuple of two str
        """
        pd_path = FileOps.join_path(
            self.trainer.local_output_path, self.trainer.step_name, "performance.csv")
        FileOps.make_base_dir(pd_path)
        df = pd.DataFrame(
            [[self.model.encoding, self.flops_count, self.params_count, self.metric]],
            columns=["encoding", "flops", "parameters", self.cfg.get("valid_metric", "acc")])
        if not os.path.exists(pd_path):
            with open(pd_path, "w") as file:
                df.to_csv(file, index=False)
        else:
            with open(pd_path, "a") as file:
                df.to_csv(file, index=False, header=False)
        if self.trainer.backup_base_path is not None:
            FileOps.copy_folder(self.trainer.local_output_path,
                                self.trainer.backup_base_path)

    def validate(self):
        """Check whether the model fits in the #flops range or #parameter range specified in config.

        :return: true or false, which specifies whether the model fits in the range
        :rtype: bool
        """
        limits_config = self.cfg.get("limits", dict())
        if "flop_range" in limits_config:
            flop_range = limits_config["flop_range"]
            if self.flops_count < flop_range[0] or self.flops_count > flop_range[1]:
                return False
        if "param_range" in limits_config:
            param_range = limits_config["param_range"]
            if self.params_count < param_range[0] or self.params_count > param_range[1]:
                return False
        return True

    def _save_best_model(self):
        save_path = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.step_name, "best_model.pth")
        FileOps.make_base_dir(save_path)
        torch.save(self.model.state_dict(), save_path)
        if self.trainer.backup_base_path is not None:
            _dst = FileOps.join_path(
                self.trainer.backup_base_path, "workers", str(self.trainer.worker_id))
            FileOps.copy_folder(self.trainer.get_local_worker_path(), _dst)
