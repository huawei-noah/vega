# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModelCheckpoint callback defination."""
import pickle
import logging
import vega
from .callback import Callback
from vega.core.common.file_ops import FileOps
from vega.core.common.class_factory import ClassFactory, ClassType

if vega.is_torch_backend():
    import torch


@ClassFactory.register(ClassType.CALLBACK)
class ModelCheckpoint(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(Callback, self).__init__()
        self.priority = 240

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.is_chief = self.params['is_chief']

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if self.is_chief and logs.get('summary_perfs').get('best_valid_perfs_changed', False):
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """Save checkpoint."""
        logging.debug("Start Save Checkpoint, file_name=%s", self.trainer.checkpoint_file_name)
        checkpoint_file = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.checkpoint_file_name)
        logging.debug("Start Save Model, model_file=%s", self.trainer.model_pickle_file_name)
        model_pickle_file = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.model_pickle_file_name)
        # pickle model
        with open(model_pickle_file, 'wb') as handle:
            pickle.dump(self.trainer.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'weight': self.trainer.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'lr_scheduler': self.trainer.lr_scheduler.state_dict(),
        }
        torch.save(ckpt, checkpoint_file)
        self.trainer.checkpoint_file = checkpoint_file
        self.trainer.model_path = model_pickle_file

    def after_train(self, logs=None):
        """Be called after the training process."""
        torch.save(self.trainer.model.state_dict(), self.trainer.weights_file)
