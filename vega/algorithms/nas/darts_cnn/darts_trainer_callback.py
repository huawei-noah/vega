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

"""DARTS trainer."""
import logging
import os
from copy import deepcopy
import vega
from vega.common import Config, FileOps
from vega.algorithms.nas.darts_cnn import DartsNetworkTemplateConfig
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks import Callback
from vega.core.search_space import SearchSpace
from vega.core.search_algs import SearchAlgorithm
from vega.trainer.modules.optimizer import Optimizer
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.modules.loss import Loss

if vega.is_torch_backend():
    pass
elif vega.is_tf_backend():
    import tensorflow as tf


@ClassFactory.register(ClassType.CALLBACK)
class DartsTrainerCallback(Callback):
    """A special callback for DartsTrainer."""

    disable_callbacks = ["ModelCheckpoint"]

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        self.unrolled = self.trainer.config.unrolled
        self.device = vega.is_gpu_device() if vega.is_gpu_device() is not True else 0
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.loss = self.trainer.loss
        self.search_alg = SearchAlgorithm(SearchSpace())
        self._set_algorithm_model(self.model)
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        if vega.is_torch_backend():
            self.valid_loader_iter = iter(self.trainer.valid_loader)

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""
        train_batch = logs['train_batch']
        train_input, train_target = train_batch
        try:
            valid_input, valid_target = next(self.valid_loader_iter)
        except Exception:
            self.valid_loader_iter = iter(self.trainer.valid_loader)
            valid_input, valid_target = next(self.valid_loader_iter)
        if vega.is_npu_device():
            valid_input, valid_target = valid_input.to(int(self.device)), valid_target.to(int(self.device))
        else:
            valid_input, valid_target = valid_input.to(self.device), valid_target.to(self.device)
        self._train_arch_step(train_input, train_target, valid_input, valid_target)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        child_desc_temp = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        logging.info('normal = %s', child_desc_temp[0])
        logging.info('reduce = %s', child_desc_temp[1])
        self._save_descript()

    def after_train(self, logs=None):
        """Be called after Training."""
        self.trainer._backup()

    def _train_arch_step(self, train_input, train_target, valid_input, valid_target):
        lr = self.lr_scheduler.get_lr()[0]
        self.search_alg.step(train_input, train_target, valid_input, valid_target,
                             lr, self.optimizer, self.loss, self.unrolled)

    def _set_algorithm_model(self, model):
        self.search_alg.set_model(model)

    def train_input_fn(self):
        """Input function for search."""

        def map_to_dict(td, vd):
            return {'train': td[0], 'valid': vd[0]}, {'train': td[1], 'valid': vd[1]}

        dataset = tf.data.Dataset.zip((self.trainer.train_loader.input_fn(),
                                       self.trainer.valid_loader.input_fn()))
        dataset = dataset.map(lambda td, vd: map_to_dict(td, vd))
        return dataset

    def model_fn(self, features, labels, mode):
        """Darts model_fn used by TensorFlow Estimator."""
        logging.info('Darts model function action')
        global_step = tf.compat.v1.train.get_global_step()
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            features, valid_features = features['train'], features['valid']
            labels, valid_labels = labels['train'], labels['valid']
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.trainer.train_loader), tf.float32)
            self.trainer.optimizer = Optimizer()(distributed=self.trainer.horovod)
            self.trainer.lr_scheduler = LrScheduler()(self.trainer.optimizer)
            self.trainer.lr_scheduler.step(epoch)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            arch_minimize_op = self.search_alg.step(valid_x=valid_features,
                                                    valid_y=valid_labels,
                                                    lr=self.trainer.lr_scheduler.get_lr()[0])
            train_op = tf.group(arch_minimize_op, update_ops)
        self.model.training = mode == tf.estimator.ModeKeys.TRAIN
        logits = self.model(features)
        logits = tf.cast(logits, tf.float32)
        self.trainer.loss = Loss()()
        loss = self.trainer.loss(logits=logits, labels=labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.control_dependencies([train_op]):
                weight_ops = self.model.get_weight_ops()
                loss_scale = self.trainer.config.loss_scale if self.trainer.use_amp else 1
                train_op = self.trainer.optimizer.step(loss, loss_scale, global_step, weight_ops)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.trainer.valid_metrics(logits, labels)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def _get_arch_weights(self):
        if vega.is_torch_backend():
            arch_weights = self.model.arch_weights
        elif vega.is_tf_backend():
            sess_config = self.trainer._init_session_config()
            with tf.compat.v1.Session(config=sess_config) as sess:
                checkpoint_file = tf.train.latest_checkpoint(self.trainer.get_local_worker_path())
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                sess.run(tf.global_variables_initializer())
                arch_weights = self.model.arch_weights
                arch_weights = [weight.eval() for weight in arch_weights]
        return arch_weights

    def _save_descript(self):
        """Save result descript."""
        template_file = self.config.darts_template_file
        genotypes = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        if template_file == "{default_darts_cifar10_template}":
            template = DartsNetworkTemplateConfig.cifar10
        elif template_file == "{default_darts_cifar100_template}":
            template = DartsNetworkTemplateConfig.cifar100
        elif template_file == "{default_darts_imagenet_template}":
            template = DartsNetworkTemplateConfig.imagenet
        else:
            dst = FileOps.join_path(self.trainer.get_local_worker_path(), os.path.basename(template_file))
            FileOps.copy_file(template_file, dst)
            template = Config(dst)
        model_desc = self._gen_model_desc(genotypes, template)
        self.trainer.config.codec = model_desc

    def _gen_model_desc(self, genotypes, template):
        model_desc = deepcopy(template)
        model_desc.super_network.cells.normal.genotype = genotypes[0]
        model_desc.super_network.cells.reduce.genotype = genotypes[1]
        return model_desc
