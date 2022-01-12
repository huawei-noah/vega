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

"""The trainer program for Auto Lane."""

import logging
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoLaneTrainerCallback(Callback):
    """Construct the trainer of Auto Lane."""

    disable_callbacks = ['ProgressLogger', 'MetricsEvaluator', "ModelStatistics"]

    def logger_patch(self):
        """Patch the default logger."""
        worker_path = self.trainer.get_local_worker_path()
        worker_spec_log_file = FileOps.join_path(worker_path, 'current_worker.log')
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)
        for hdlr in logging.root.handlers:
            logging.root.removeHandler(hdlr)
        logger.addHandler(logging.FileHandler(worker_spec_log_file))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logging.root = logger

    def before_train(self, logs=None):
        """Be called before the whole train process."""
        self.trainer.config.call_metrics_on_train = False
        self.cfg = self.trainer.config
        self.worker_id = self.trainer.worker_id
        self.local_base_path = self.trainer.local_base_path
        self.local_output_path = self.trainer.local_output_path

        self.result_path = FileOps.join_path(self.trainer.local_base_path, "result")
        FileOps.make_dir(self.result_path)
        self.logger_patch()

    def make_batch(self, batch):
        """Make batch for each training step."""
        image = batch.pop('image').cuda(non_blocking=True).float()
        return image, batch

    def train_step(self, batch):
        """Replace the default train_step function."""
        self.trainer.model.train()
        image, train_item_spec = batch

        gt_loc = train_item_spec.pop('gt_loc').cuda(non_blocking=True).float()
        gt_cls = train_item_spec.pop('gt_cls').cuda(non_blocking=True).float()
        self.trainer.optimizer.zero_grad()
        model_out = self.trainer.model(input=image,
                                       gt_loc=gt_loc,
                                       gt_cls=gt_cls,
                                       forward_switch='train',
                                       **train_item_spec)
        loss_pos = model_out['loss_pos']
        loss_neg = model_out['loss_neg']
        loss_loc = model_out['loss_loc']
        loss = loss_loc + loss_pos + loss_neg
        if self.trainer.use_amp:
            raise NotImplementedError('Amp is not implemented in algorithm auto lane.')
        loss.backward()
        self.trainer.optimizer.step()
        return {'loss': loss.item(),
                'cls_pos_loss': loss_pos.item(),
                'cls_neg_loss': loss_neg.item(),
                'loc_loss': loss_loc.item(),
                'train_batch_output': None}

    def valid_step(self, batch):
        """Be called on each batch validing."""
        self.trainer.model.eval()
        image, valid_item_spec = batch
        results = self.trainer.model(input=image,
                                     forward_switch='valid',
                                     **valid_item_spec)
        return {'valid_batch_output': results}
