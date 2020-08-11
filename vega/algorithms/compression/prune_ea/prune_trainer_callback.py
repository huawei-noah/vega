# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer for searching pruned model."""
import copy
import os
import numpy as np
import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.core.metrics import calc_model_flops_params
from vega.core.trainer.callbacks import Callback
from vega.search_space.networks import NetworkDesc

if vega.is_torch_backend():
    import torch
    import torch.nn as nn
elif vega.is_tf_backend():
    import tensorflow as tf


@ClassFactory.register(ClassType.CALLBACK)
class PruneTrainerCallback(Callback):
    """Callback of Prune Trainer."""

    disable_callbacks = ["ModelStatistics"]

    def __init__(self):
        super(Callback, self).__init__()
        self.flops_count = None
        self.params_count = None

    def before_train(self, logs=None):
        """Be called before the train process."""
        self.config = self.trainer.config
        self.device = self.trainer.config.device
        self.base_net_desc = self.trainer.config.codec
        if vega.is_torch_backend():
            self.trainer.model._init_weights()
            count_input = torch.FloatTensor(1, 3, 32, 32).to(self.device)
        elif vega.is_tf_backend():
            tf.reset_default_graph()
            count_input = tf.random_uniform([1, 32, 32, 3], dtype=tf.float32)
        self.flops_count, self.params_count = calc_model_flops_params(self.trainer.model, count_input)
        self.validate()
        self.trainer.model = self._generate_init_model(self.trainer.model)

    def after_epoch(self, epoch, logs=None):
        """Update gflops and kparams."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'gflops': self.flops_count, 'kparams': self.params_count})
        logs.update({'summary_perfs': summary_perfs})

    def _new_model_init(self, model_prune):
        """Init new model.

        :param model_prune: searched pruned model
        :type model_prune: torch.nn.Module
        :return: initial model after loading pretrained model
        :rtype: torch.nn.Module
        """
        init_model_file = self.config.init_model_file
        if ":" in init_model_file:
            local_path = FileOps.join_path(
                self.trainer.get_local_worker_path(), os.path.basename(init_model_file))
            FileOps.copy_file(init_model_file, local_path)
            self.config.init_model_file = local_path
        network_desc = copy.deepcopy(self.base_net_desc)
        network_desc.backbone.chn = network_desc.backbone.base_chn
        network_desc.backbone.chn_node = network_desc.backbone.base_chn_node
        network_desc.backbone.encoding = model_prune.encoding
        model_init = NetworkDesc(network_desc).to_model()
        return model_init

    def _init_chn_node_mask(self, model_prune):
        """Init channel node mask.

        :param model_prune: searched pruned model
        :type model_prune: torch.nn.Module
        :return: channel node masks
        :rtype: array
        """
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
        model_init = self._new_model_init(model_prune)
        chn_node_mask = self._init_chn_node_mask(model_prune)
        if vega.is_torch_backend():
            return self._load_torch_model(model_prune, model_init, chn_node_mask)
        elif vega.is_tf_backend():
            return self._load_tf_model(model_prune, model_init, chn_node_mask)

    def validate(self):
        """Check whether the model fits in the #flops range or #parameter range specified in config.

        :return: true or false, which specifies whether the model fits in the range
        :rtype: bool
        """
        limits_config = self.config.limits or dict()
        if "flop_range" in limits_config:
            flop_range = limits_config["flop_range"]
            if self.flops_count < flop_range[0] or self.flops_count > flop_range[1]:
                raise ValueError("flop count exceed limits range.")
        if "param_range" in limits_config:
            param_range = limits_config["param_range"]
            if self.params_count < param_range[0] or self.params_count > param_range[1]:
                raise ValueError("params count exceed limits range.")
        return True

    def _load_torch_model(self, model_prune, model_init, chn_node_mask):
        """Load torch pretrained model."""
        checkpoint = torch.load(self.config.init_model_file)
        model_init.load_state_dict(checkpoint)
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
                        layer1.weight.data = layer1.weight.data * torch.FloatTensor(mask)
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
                        layer1.weight.data = layer1.weight.data * torch.FloatTensor(mask)
                        layer1.bias.data = layer1.bias.data * torch.FloatTensor(mask)
                        layer1.running_mean = layer1.running_mean * torch.FloatTensor(mask)
                        layer1.running_var = layer1.running_var * torch.FloatTensor(mask)
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
        model_init.to(self.device)
        return model_init

    def _load_tf_model(self, model_prune, model_init, chn_node_mask):
        """Load tensorflow pretrained model."""
        with tf.Session(config=self.trainer._init_session_config()) as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(self.config.init_model_file))
            saver.restore(sess, self.config.init_model_file)
            chn_node_id = 0
            chn_id = 0
            chn_mask = model_prune.chn_mask
            start_mask = []
            end_mask = []

            all_weight = tf.get_collection(tf.GraphKeys.VARIABLES)
            all_weight = [t for t in all_weight if not t.name.endswith('Momentum:0')]
            for op in all_weight:
                name = op.name
                if name.startswith('conv_1'):
                    end_mask = chn_node_mask[0]
                    end_mask = np.asarray(end_mask)
                    idx1 = np.squeeze(np.argwhere(np.asarray(
                        np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(op.get_shape())
                    mask[:, :, :, idx1.tolist()] = 0
                    sess.run(tf.assign(op, op * tf.constant(mask, dtype=op.dtype)))
                elif name.startswith('bn_1'):
                    idx1 = np.squeeze(np.argwhere(np.asarray(
                        np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(op.get_shape())
                    mask[idx1.tolist()] = 0
                    sess.run(tf.assign(op, op * tf.constant(mask, dtype=op.dtype)))
                elif name.startswith('dense/kernel'):
                    idx1 = np.squeeze(np.argwhere(
                        np.asarray(np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(op.get_shape())
                    mask[idx1.tolist(), :] = 0
                    sess.run(tf.assign(op, op * tf.constant(mask, dtype=op.dtype)))
                elif name.startswith('layer'):
                    parsed_name = list(name.split('/'))
                    layer_idx = parsed_name[0][-1]
                    block_idx = parsed_name[1][-1]
                    operation = parsed_name[2]
                    if operation.startswith('conv'):
                        if operation == 'conv_1':
                            start_mask = chn_node_mask[int(layer_idx) - 1]
                            end_mask = chn_mask[int(block_idx)]
                        if operation == 'conv_2':
                            start_mask = end_mask
                            end_mask = chn_node_mask[int(layer_idx)]
                        # shortcut
                        if operation == 'conv_3':
                            start_mask = chn_node_mask[int(layer_idx) - 1]
                            end_mask = chn_node_mask[int(layer_idx)]
                        start_mask = np.asarray(start_mask)
                        end_mask = np.asarray(end_mask)
                        idx0 = np.squeeze(np.argwhere(
                            np.asarray(np.ones(start_mask.shape) - start_mask)))
                        idx1 = np.squeeze(np.argwhere(
                            np.asarray(np.ones(end_mask.shape) - end_mask)))
                        mask = np.ones(op.get_shape())
                        mask[:, :, idx0.tolist(), :] = 0
                        mask[:, :, :, idx1.tolist()] = 0
                        sess.run(tf.assign(op, op * tf.constant(mask, dtype=op.dtype)))
                    elif operation.startswith('bn'):
                        idx1 = np.squeeze(np.argwhere(np.asarray(
                            np.ones(end_mask.shape) - end_mask)))
                        mask = np.ones(op.get_shape())
                        mask[idx1.tolist()] = 0
                        sess.run(tf.assign(op, op * tf.constant(mask, dtype=op.dtype)))
            save_file = FileOps.join_path(self.trainer.get_local_worker_path(), 'prune_model')
            saver.save(sess, save_file)
            return model_init
