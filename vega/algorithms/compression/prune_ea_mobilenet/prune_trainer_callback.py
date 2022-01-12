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
import copy
import os
import vega
from vega.algorithms.compression.prune_ea.prune_trainer_callback import PruneTrainerCallback
from vega.common import ClassFactory, ClassType, FileOps
from vega.networks.network_desc import NetworkDesc
from vega.modules.operators import PruneMobileNet


@ClassFactory.register(ClassType.CALLBACK)
class PruneMobilenetTrainerCallback(PruneTrainerCallback):
    """Callback of Prune Trainer."""

    def __init__(self):
        super(PruneMobilenetTrainerCallback, self).__init__()

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
        network_desc.backbone.cfgs = network_desc.backbone.base_cfgs
        model_init = NetworkDesc(network_desc).to_model()
        return model_init

    def _init_chn_node_mask(self):
        """Init channel node mask.

        :return: channel node masks
        :rtype: array
        """
        chn_mask = self.base_net_desc.backbone.chn_mask
        return chn_mask

    def _generate_init_model(self):
        """Generate init model by loading pretrained model.

        :return: initial model after loading pretrained model
        :rtype: torch.nn.Module
        """
        model_init = self._new_model_init()
        chn_mask = self._init_chn_node_mask()
        if vega.is_torch_backend():
            import torch
            checkpoint = torch.load(self.config.init_model_file + '.pth')
            model_init.load_state_dict(checkpoint)
            model = PruneMobileNet(model_init).apply(chn_mask)
            model.to(self.device)
        elif vega.is_tf_backend():
            import tensorflow as tf
            model = model_init
            with tf.compat.v1.Session(config=self.trainer._init_session_config()) as sess:
                saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(self.config.init_model_file))
                saver.restore(sess, self.config.init_model_file)
                all_weight = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.VARIABLES)
                all_weight = [t for t in all_weight if not t.name.endswith('Momentum:0')]
                PruneMobileNet(all_weight).apply(chn_mask)
                save_file = FileOps.join_path(self.trainer.get_local_worker_path(), 'prune_model')
                saver.save(sess, save_file)
        elif vega.is_ms_backend():
            from mindspore.train.serialization import load_checkpoint, load_param_into_net
            parameter_dict = load_checkpoint(self.config.init_model_file)
            load_param_into_net(model_init, parameter_dict)
            model = PruneMobileNet(model_init).apply(chn_mask)
        return model
