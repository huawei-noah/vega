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

"""Trainer for searching pruned model."""
import logging
import copy
import os
import vega
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.metrics import calc_model_flops_params, calc_forward_latency
from vega.trainer.callbacks import Callback
from vega.networks.network_desc import NetworkDesc
from vega.modules.operators import PruneResnet
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.trainer.modules.optimizer import Optimizer
import numpy as np

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf
elif vega.is_ms_backend():
    import mindspore
    from mindspore.train.serialization import load_checkpoint, load_param_into_net


@ClassFactory.register(ClassType.CALLBACK)
class PruneTrainerCallback(Callback):
    """Callback of Prune Trainer."""

    disable_callbacks = ["ModelStatistics"]

    def __init__(self):
        super(Callback, self).__init__()
        self.flops_count = None
        self.params_count = None
        self.latency_count = None

    def before_train(self, logs=None):
        """Be called before the train process."""
        self.config = self.trainer.config
        self.device = vega.is_gpu_device() if vega.is_gpu_device() is not True else 0
        self.base_net_desc = self.trainer.model_desc
        sess_config = None
        if vega.is_torch_backend():
            if vega.is_npu_device():
                count_input = torch.FloatTensor(1, 3, 32, 32).to(vega.get_devices())
            elif vega.is_gpu_device():
                count_input = torch.FloatTensor(1, 3, 32, 32).to(self.device)
        elif vega.is_tf_backend():
            count_input = tf.random.uniform([1, 3, 32, 32], dtype=tf.float32)
            sess_config = self.trainer._init_session_config()
        elif vega.is_ms_backend():
            count_input = mindspore.Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        self.flops_count, self.params_count = calc_model_flops_params(self.trainer.model, count_input)
        self.latency_count = calc_forward_latency(self.trainer.model, count_input, sess_config)
        logging.info("after prune model glops=%sM, params=%sK, latency=%sms",
                     self.flops_count * 1e-6, self.params_count * 1e-3, self.latency_count * 1000)
        self.trainer.model = self._generate_init_model()
        if vega.is_torch_backend():
            self.trainer.optimizer = Optimizer()(model=self.trainer.model, distributed=self.trainer.horovod)
            self.trainer.lr_scheduler = LrScheduler()(self.trainer.optimizer)

    def after_epoch(self, epoch, logs=None):
        """Update flops and params."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'flops': self.flops_count * 1e-6, 'params': self.params_count * 1e-3,
                              'latency': self.latency_count * 1000})
        logs.update({'summary_perfs': summary_perfs})

    def _new_model_init(self):
        """Init new model.

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
        network_desc.backbone.base_channel = network_desc.backbone.base_chn_node[0]
        network_desc.head.base_channel = network_desc.backbone.base_chn[-1]
        model_init = NetworkDesc(network_desc).to_model()
        return model_init

    def _init_chn_node_mask(self):
        """Init channel node mask.

        :return: channel node masks
        :rtype: array
        """
        chn_node_mask_tmp = self.base_net_desc.backbone.chn_node_mask
        chn_node_mask = [single_mask for (i, single_mask) in zip([1, 3, 3, 3], chn_node_mask_tmp)
                         for _ in range(i)]
        return chn_node_mask

    def _generate_init_model(self):
        """Generate init model by loading pretrained model.

        :return: initial model after loading pretrained model
        :rtype: torch.nn.Module
        """
        model_init = self._new_model_init()
        chn_node_mask = self._init_chn_node_mask()
        if vega.is_torch_backend():
            if vega.is_gpu_device():
                checkpoint = torch.load(self.config.init_model_file + '.pth')
                model_init.load_state_dict(checkpoint)
                model = PruneResnet(model_init).apply(chn_node_mask, self.base_net_desc.backbone.chn_mask)
                model.to(self.device)
            elif vega.is_npu_device():
                device = "npu:{}".format(os.environ.get('DEVICE_ID', 0))
                checkpoint = torch.load(self.config.init_model_file + '.pth',
                                        map_location=torch.device('{}'.format(device)))
                model_init.load_state_dict(checkpoint)
                model = PruneResnet(model_init).apply(chn_node_mask, self.base_net_desc.backbone.chn_mask)
                model.to(vega.get_devices())
        elif vega.is_tf_backend():
            model = model_init
            with tf.compat.v1.Session(config=self.trainer._init_session_config()) as sess:
                saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(self.config.init_model_file))
                saver.restore(sess, self.config.init_model_file)
                all_weight = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.VARIABLES)
                all_weight = [t for t in all_weight if not t.name.endswith('Momentum:0')]
                PruneResnet(all_weight).apply(chn_node_mask, self.base_net_desc.backbone.chn_mask)
                save_file = FileOps.join_path(self.trainer.get_local_worker_path(), 'prune_model')
                saver.save(sess, save_file)
        elif vega.is_ms_backend():
            parameter_dict = load_checkpoint(self.config.init_model_file)
            load_param_into_net(model_init, parameter_dict)
            model = PruneResnet(model_init).apply(chn_node_mask, self.base_net_desc.backbone.chn_mask)
        return model
