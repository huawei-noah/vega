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

"""The trainer program for Adelaide_EA."""
import logging
import numpy as np
import vega
from vega.common import ClassFactory, ClassType, General
from vega.metrics import calc_model_flops_params
from vega.trainer.callbacks import Callback

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf
elif vega.is_ms_backend():
    import mindspore
    from mindspore.train import Model as MsModel

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AdelaideEATrainerCallback(Callback):
    """Construct the trainer of Adelaide-EA."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        input_shape = [1, 3, 192, 192] if General.data_format == 'channels_first' else [1, 192, 192, 3]
        if vega.is_torch_backend():
            if vega.is_gpu_device():
                count_input = torch.FloatTensor(*input_shape).cuda()
            elif vega.is_npu_device():
                input_shape = [1, 3, 192, 192]
                count_input = torch.FloatTensor(*input_shape).to(vega.get_devices())
        elif vega.is_tf_backend():
            tf.compat.v1.reset_default_graph()
            count_input = tf.random.uniform(input_shape, dtype=tf.float32)
        elif vega.is_ms_backend():
            count_input = mindspore.Tensor(np.random.randn(*input_shape).astype(np.float32))
            loss_fn = ClassFactory.get_cls(ClassType.LOSS, "CustomSoftmaxCrossEntropyWithLogits")()
            self.trainer.ms_model = MsModel(network=self.trainer.model,
                                            loss_fn=loss_fn,
                                            optimizer=self.trainer.optimizer,
                                            metrics=self.trainer.ms_metrics)

        flops_count, params_count = calc_model_flops_params(self.trainer.model, count_input)
        self.flops_count, self.params_count = flops_count * 1e-9, params_count * 1e-3
        logger.info("Flops: {:.2f} G, Params: {:.1f} K".format(self.flops_count, self.params_count))

    def after_epoch(self, epoch, logs=None):
        """Update flops and params."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'flops': self.flops_count, 'params': self.params_count})
        logs.update({'summary_perfs': summary_perfs})
