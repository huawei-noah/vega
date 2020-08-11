# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""cross Entropy Weight Decay Loss."""
import importlib
import tensorflow as tf
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class CrossEntropyWeightDecay(object):
    """Loss lass of Cross Entroppy Weight Decay.

    :param cross_entropy: cross entropy function name
    :type cross_entropy: str
    :param weight_decay: weight decay for regulation
    :type weight_decay: float
    """

    def __init__(self, cross_entropy, weight_decay, ignore_index=None):
        self.cross_entropy = getattr(
            importlib.import_module('tensorflow.losses'),
            cross_entropy)
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index

    def __call__(self, logits, labels):
        """Loss forward function."""
        if self.ignore_index is not None:
            logits, labels = self.exclude_ignore_index(logits, labels)
        cross_entropy_loss = self.cross_entropy(logits=logits, labels=labels)
        l2_loss_list = [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if 'batch_normalization' not in v.name]
        loss = cross_entropy_loss + self.weight_decay * tf.add_n(l2_loss_list)
        return loss

    def exclude_ignore_index(self, logits, labels):
        """Ignore certen index."""
        logits = tf.reshape(logits, [-1, logits.shape.as_list()[-1]])
        labels = tf.reshape(labels, [-1])
        indices = tf.squeeze(tf.where(tf.not_equal(labels, self.ignore_index)), 1)
        labels = tf.cast(tf.gather(labels, indices), tf.int32)
        logits = tf.gather(logits, indices)
        return logits, labels
