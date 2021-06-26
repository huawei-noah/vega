# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TF Adam."""
import vega

if vega.is_tf_backend():
    import tensorflow as tf


class OptimizerStep(object):
    """Adam optimizer for tensorflow."""

    def __init__(self, learning_rate, weight_decay=0.):
        self.weight_decay = weight_decay
        self.base_lr = learning_rate

    def set_lr(self, learning_rate):
        """Uptate learning rate of optimizer."""
        if hasattr(self, '_learning_rate'):
            self._learning_rate = learning_rate
        elif hasattr(self, '_lr'):
            self._lr = learning_rate

    def step(self, loss, loss_scale, global_step, var_list=None):
        """Compute and update gradients."""
        loss = loss + self.regularize_loss(loss)
        if loss_scale != 1:
            scaled_grad_vars = self.compute_gradients(loss * loss_scale, var_list=var_list)
            unscaled_grad_vars = []
            for grad, var in scaled_grad_vars:
                unscaled_grad_vars.append((grad, var) if grad is None else (grad / loss_scale, var))
            minimize_op = self.apply_gradients(unscaled_grad_vars, global_step)
        else:
            grad_vars = self.compute_gradients(loss, var_list=var_list)
            minimize_op = self.apply_gradients(grad_vars, global_step)
        return minimize_op

    def regularize_loss(self, loss):
        """Compute and return l2 loss."""
        l2_loss_list = [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()
                        if 'batch_normalization' not in v.name]
        loss = loss + self.weight_decay * tf.add_n(l2_loss_list)
        return loss


def dynamic_optimizer(optimizer_class, **params):
    """Dynamically choose optimizer."""
    class DynamicOptimizer(optimizer_class, OptimizerStep):
        """Dynamic optimizer for tensorflow."""

        def __init__(self, **kwargs):
            weight_decay = 0.
            learning_rate = 0.
            if 'weight_decay' in kwargs:
                weight_decay = kwargs.pop('weight_decay')
            if 'learning_rate' in kwargs:
                learning_rate = kwargs['learning_rate']
            optimizer_class.__init__(self, **kwargs)
            OptimizerStep.__init__(self, learning_rate=learning_rate, weight_decay=weight_decay)
    return DynamicOptimizer(**params)


def dynamic_distributed_optimizer(optimizer_class, optimizer):
    """Dynamically choose distributed optimizer."""
    class DynamicDistributedOptimizer(optimizer_class, OptimizerStep):
        """Dynamic distributed optimizer for tensorflow."""

        def __init__(self, optimizer):
            optimizer_class.__init__(self, optimizer)
            OptimizerStep.__init__(self, learning_rate=optimizer.base_lr, weight_decay=optimizer.weight_decay)
    return DynamicDistributedOptimizer(optimizer)
