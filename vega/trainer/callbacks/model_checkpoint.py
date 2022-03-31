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

"""ModelCheckpoint callback defination."""

import os
import glob
import logging
import dill
import numpy as np
import vega
from vega.common import FileOps
from vega.common import ClassFactory, ClassType
from .callback import Callback

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf


@ClassFactory.register(ClassType.CALLBACK)
class ModelCheckpoint(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(ModelCheckpoint, self).__init__()
        self.priority = 240

    def before_train(self, logs=None):
        """Be called before the training process."""
        if self.trainer.load_checkpoint:
            self._load_checkpoint()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if not self.trainer.config.save_checkpoint:
            return
        if not self.trainer.do_validation:
            self._save_best_model()
            return
        self._save_checkpoint(epoch)
        if self.trainer.multi_task:
            self._saved_multi_checkpoint(epoch)
        if self.trainer.is_chief and logs.get('summary_perfs').get('best_changed', False):
            self._save_best_model()

    def _save_best_model(self):
        """Save best model."""
        if vega.is_torch_backend():
            torch.save(self.trainer.model.state_dict(), self.trainer.weights_file)
        elif vega.is_tf_backend():
            worker_path = self.trainer.get_local_worker_path()
            model_id = "model_{}".format(self.trainer.worker_id)
            weights_folder = FileOps.join_path(worker_path, model_id)
            FileOps.make_dir(weights_folder)
            checkpoint_file = tf.train.latest_checkpoint(worker_path)
            ckpt_globs = glob.glob("{}.*".format(checkpoint_file))
            for _file in ckpt_globs:
                FileOps.copy_file(_file, FileOps.join_path(weights_folder, os.path.split(_file)[-1]))
            FileOps.copy_file(FileOps.join_path(worker_path, 'checkpoint'), weights_folder)
            if self.trainer.save_ext_model:
                self._save_pb_model(weights_folder, model_id)
                self.trainer.ext_model = FileOps.join_path(weights_folder, '{}.pb'.format(model_id))
        elif vega.is_ms_backend():
            worker_path = self.trainer.get_local_worker_path()
            save_path = os.path.join(worker_path, "model_{}.ckpt".format(self.trainer.worker_id))
            for file in os.listdir(worker_path):
                if file.startswith("CKP") and file.endswith(".ckpt"):
                    self.weights_file = FileOps.join_path(worker_path, file)
                    os.rename(self.weights_file, save_path)
            if self.trainer.save_ext_model:
                model_id = "model_{}".format(self.trainer.worker_id)
                self._save_om_model(worker_path, model_id)
                self.trainer.ext_model = FileOps.join_path(worker_path, '{}.om'.format(model_id))

    def _save_checkpoint(self, epoch):
        """Save checkpoint."""
        if not self.trainer.config.save_slave_model:
            if not self.trainer.is_chief:
                return
        logging.debug("Start Save Checkpoint, file_name=%s", self.trainer.checkpoint_file_name)
        checkpoint_file = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.checkpoint_file_name)
        logging.debug("Start Save Model, model_file=%s", self.trainer.model_pickle_file_name)
        if vega.is_torch_backend():
            ckpt = {
                'epoch': epoch,
                'weight': self.trainer.model.state_dict(),
                'optimizer': self.trainer.optimizer.state_dict(),
                'lr_scheduler': self.trainer.lr_scheduler.state_dict(),
            }
            torch.save(ckpt, checkpoint_file, pickle_module=dill)
        self.trainer.checkpoint_file = checkpoint_file

    def _load_checkpoint(self):
        """Load checkpoint."""
        if vega.is_torch_backend():
            if hasattr(self.trainer.config, "checkpoint_path"):
                checkpoint_path = self.trainer.config.checkpoint_path
            else:
                checkpoint_path = self.trainer.get_local_worker_path()
            checkpoint_file = FileOps.join_path(checkpoint_path, self.trainer.checkpoint_file_name)
            if os.path.exists(checkpoint_file):
                try:
                    logging.info("Load checkpoint file, file={}".format(checkpoint_file))
                    checkpoint = torch.load(checkpoint_file, pickle_module=dill)
                    if self.trainer.multi_task:
                        self.trainer.model.load_state_dict(checkpoint["weight"], strict=False)
                        multi_task_checkpoint = torch.load(
                            FileOps.join_path(
                                checkpoint_path, self.trainer.multi_task, self.trainer.checkpoint_file_name),
                            pickle_module=dill)
                        self.trainer.optimizer.load_state_dict(multi_task_checkpoint["optimizer"])
                        self.trainer.lr_scheduler.load_state_dict(multi_task_checkpoint["lr_scheduler"])
                    else:
                        self.trainer.model.load_state_dict(checkpoint["weight"])
                        self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                    if self.trainer._resume_training:
                        self.trainer._start_epoch = checkpoint["epoch"] + 1
                        logging.info("Resume fully train, change start epoch to {}".format(self.trainer._start_epoch))
                except Exception as e:
                    logging.info("Load checkpoint failed {}".format(e))
            else:
                logging.info("skip loading checkpoint file that do not exist, {}".format(checkpoint_file))

    def _saved_multi_checkpoint(self, epoch):
        """Save multi tasks checkpoint."""
        FileOps.make_dir(self.trainer.get_local_worker_path(), self.trainer.multi_task)
        checkpoint_file = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.multi_task, self.trainer.checkpoint_file_name)
        logging.debug("Start Save Multi Task Model, model_file=%s", self.trainer.model_pickle_file_name)
        if vega.is_torch_backend():
            ckpt = {
                'epoch': epoch,
                'weight': self.trainer.model.state_dict(),
                'optimizer': self.trainer.optimizer.state_dict(),
                'lr_scheduler': self.trainer.lr_scheduler.state_dict(),
            }
            torch.save(ckpt, checkpoint_file, pickle_module=dill)
        self.trainer.checkpoint_file = checkpoint_file

    def _save_pb_model(self, weight_file, model_id):
        from tensorflow.python.framework import graph_util
        valid_data = self.trainer.valid_loader.input_fn()
        iterator = valid_data.make_one_shot_iterator()
        one_element = iterator.get_next()
        with tf.Session() as sess:
            batch = sess.run(one_element)
        input_shape = batch[0].shape
        with tf.Graph().as_default():
            input_holder_shape = (None,) + tuple(input_shape[1:])
            input_holder = tf.placeholder(dtype=tf.float32, shape=input_holder_shape)
            self.trainer.model.training = False
            output = self.trainer.model(input_holder)
            if isinstance(output, tuple):
                output_name = [output[0].name.split(":")[0]]
            else:
                output_name = [output.name.split(":")[0]]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if weight_file is not None:
                    saver = tf.train.Saver()
                    last_weight_file = tf.train.latest_checkpoint(weight_file)
                    if last_weight_file:
                        saver.restore(sess, last_weight_file)
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_name)
                output_graph = FileOps.join_path(weight_file, '{}.pb'.format(model_id))
                with tf.gfile.FastGFile(output_graph, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

    def _save_om_model(self, weight_file, model_id):
        from mindspore.train.serialization import export
        from mindspore import Tensor
        import subprocess
        for _, batch in enumerate(self.trainer.valid_loader.create_dict_iterator()):
            data = batch["image"]
        input_shape = data.shape
        fake_input = np.random.random(input_shape).astype(np.float32)
        save_name = os.path.join(weight_file, "{}".format(model_id))
        export(self.trainer.model, Tensor(fake_input), file_name=save_name, file_format='GEIR')
        try:
            p = subprocess.Popen(
                ["atc", "--model={}.air".format(save_name), "--output={}".format(save_name), "--soc_version=Ascend310",
                 "--framework=1", "--core_type=AiCore", "--disable_reuse_memory=1"],
                env=os.environ)
            p.wait()
        except Exception as e:
            raise ValueError('Convert model failed.Error: {}'.format(e))
