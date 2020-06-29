# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Callbacks called at certain points of trainer."""

from vega.core.common.class_factory import ClassFactory, ClassType


class CallbackList(object):
    """A container for managing registered Callback Objects."""

    def __init__(self, callbacks=None):
        """Init Callback container."""
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.trainer = None
        self.make_batch = None
        self.train_step = None
        self.valid_step = None
        self.params = {}
        for callback in self.callbacks:
            # Get make_batch if callback has defined one
            if type(callback).make_batch != Callback.make_batch:
                if self.make_batch is None:
                    self.make_batch = callback.make_batch
                else:
                    raise ValueError("Multiple make_batch are defined!")
            # Get train_step if callback has defined one
            if type(callback).train_step != Callback.train_step:
                if self.train_step is None:
                    self.train_step = callback.train_step
                else:
                    raise ValueError("Multiple train_step are defined!")
            # Get valid_step if callback has defined one
            if type(callback).valid_step != Callback.valid_step:
                if self.valid_step is None:
                    self.valid_step = callback.valid_step
                else:
                    raise ValueError("Multiple valid_step tare defined!")

    def set_params(self, params):
        """Set the trainer object for callback container."""
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        """Set the trainer object for callback container."""
        self.trainer = trainer
        # Replace the default make_batch of Trainer
        if self.make_batch is not None:
            self.trainer.make_batch = self.make_batch
        # Replace the default train_step of Trainer
        if self.train_step is not None:
            self.trainer.train_step = self.train_step
        # Replace the default train_step of Trainer
        if self.valid_step is not None:
            self.trainer.valid_step = self.valid_step

        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def before_train(self, logs=None):
        """Call before_train of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_train(logs)

    def before_epoch(self, epoch, logs=None):
        """Call before_epoch of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_epoch(epoch, logs)

    def before_train_step(self, batch_index, logs=None):
        """Call before_train_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_train_step(batch_index, logs)

    def after_train_step(self, batch_index, logs=None):
        """Call after_train_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_train_step(batch_index, logs)

    def after_epoch(self, epoch, logs=None):
        """Call after_epoch of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_epoch(epoch, logs)

    def after_train(self, logs=None):
        """Call after_train of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_train(logs)

    def before_valid(self, logs=None):
        """Call before_valid of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_valid(logs)

    def before_valid_step(self, batch_index, logs=None):
        """Call before_valid_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_valid_step(batch_index, logs)

    def after_valid_step(self, batch_index, logs=None):
        """Call after_valid_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_valid_step(batch_index, logs)

    def after_valid(self, logs=None):
        """Call after_valid of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_valid(logs)


@ClassFactory.register(ClassType.CALLBACK)
class Callback(object):
    """Abstract class for buiding new callbacks."""

    def __init__(self):
        """Init callback object."""
        self.trainer = None
        self.params = None

    def set_trainer(self, trainer):
        """Set trainer object for current callback."""
        self.trainer = trainer

    def set_params(self, params):
        """Set parameters for current callback."""
        self.params = params

    def before_train(self, logs=None):
        """Be called before the training process.

        Subclasses should override this for their own purposes
        """

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch during the training process.

        Subclasses should override this for their own purposes
        """

    def before_train_step(self, batch_index, logs=None):
        """Be called before each batch training.

        Subclasses should override this for their own purposes
        """

    def make_batch(self, batch):
        """Be called on each batch training.

        Subclasses should override this for their own purposes
        This will replace the default make_batch function in the
        trainer.
        """

    def train_step(self, batch):
        """Be called on each batch training.

        Subclasses should override this for their own purposes
        This will replace the default train_step function in the
        trainer.
        """

    def valid_step(self, batch):
        """Be called on each batch validing.

        Subclasses should override this for their own purposes
        This will replace the default valid_step function in the
        valider.
        """

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch training.

        Subclasses should override this for their own purposes
        """

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch during the training process.

        Subclasses should override this for their own purposes
        """

    def after_train(self, logs=None):
        """Be called after the training process.

        Subclasses should override this for their own purposes
        """

    def before_valid(self, logs=None):
        """Be called before the validation.

        Subclasses should override this for their own purposes

        Also called before a validation batch during the train function
        """

    def before_valid_step(self, batch_index, logs=None):
        """Be called before a batch evaluation or validation.

        Subclasses should override this for their own purposes

        Also called before a validation batch during the train function
        if validition is requied
        """

    def after_valid_step(self, batch_index, logs=None):
        """Be called after a batch validation.

        Subclasses should override this for their own purposes

        Also called after a validation batch during the train function,
        if validition is requied
        """

    def after_valid(self, logs=None):
        """Be called after the validation.

        Subclasses should override this for their own purposes
        """
