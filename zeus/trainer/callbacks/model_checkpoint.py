# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModelCheckpoint callback defination."""
import os
import logging
import zeus
from .callback import Callback
from zeus.common import FileOps
from zeus.common import ClassFactory, ClassType

if zeus.is_torch_backend():
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
        if self.trainer.load_checkpoint:
            self._load_checkpoint()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self._save_checkpoint(epoch)
        if self.is_chief and logs.get('summary_perfs').get('best_valid_perfs_changed', False):
            self._save_best_model()

    def _save_best_model(self):
        """Save best model."""
        if zeus.is_torch_backend():
            torch.save(self.trainer.model.state_dict(), self.trainer.weights_file)

    def _save_checkpoint(self, epoch):
        """Save checkpoint."""
        logging.debug("Start Save Checkpoint, file_name=%s", self.trainer.checkpoint_file_name)
        checkpoint_file = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.checkpoint_file_name)
        logging.debug("Start Save Model, model_file=%s", self.trainer.model_pickle_file_name)
        # save checkpoint
        if zeus.is_torch_backend():
            ckpt = {
                'epoch': epoch,
                'weight': self.trainer.model.state_dict(),
                'optimizer': self.trainer.optimizer.state_dict(),
                'lr_scheduler': self.trainer.lr_scheduler.state_dict(),
            }
            torch.save(ckpt, checkpoint_file)
        self.trainer.checkpoint_file = checkpoint_file

    def _load_checkpoint(self):
        """Load checkpoint."""
        if zeus.is_torch_backend():
            checkpoint_file = FileOps.join_path(
                self.trainer.get_local_worker_path(), self.trainer.checkpoint_file_name)
            if os.path.exists(checkpoint_file):
                try:
                    logging.info("Load checkpoint file, file={}".format(checkpoint_file))
                    checkpoint = torch.load(checkpoint_file)
                    self.trainer.model.load_state_dict(checkpoint["weight"])
                    self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                    if self.trainer._resume_training:
                        epoch = checkpoint["epoch"]
                        self.trainer._start_epoch = checkpoint["epoch"]
                        logging.info("Resume fully train, change start epoch to {}".format(self.trainer._start_epoch))
                except Exception as e:
                    logging.info("Load checkpoint failed {}".format(e))
            else:
                logging.info('Use default model')
