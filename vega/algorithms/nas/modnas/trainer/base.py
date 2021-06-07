# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base Trainer."""
from modnas.utils.logging import get_logger
from modnas.utils import DummyWriter
from modnas.core.event import event_hooked_subclass


@event_hooked_subclass
class TrainerBase():
    """Base Trainer class."""

    logger = get_logger('trainer')

    def __init__(self, writer=None):
        if writer is None:
            writer = DummyWriter()
        self.writer = writer

    def init(self, model, config=None):
        """Initialize trainer states."""
        raise NotImplementedError

    def model_input(self, data):
        """Return model input."""
        return data[:-1], {}

    def model_output(self, *args, data=None, model=None, attr=None, **kwargs):
        """Return model output."""
        model_fn = model if attr is None else getattr(model, attr)
        if data is not None:
            args, kwargs = self.model_input(data)
        return model_fn(*args, **kwargs)

    def loss(self, output=None, data=None, model=None):
        """Return loss."""
        return None

    def train_epoch(self):
        """Train for one epoch."""
        raise NotImplementedError

    def valid_epoch(self):
        """Validate for one epoch."""
        raise NotImplementedError

    def train_step(self):
        """Train for one step."""
        raise NotImplementedError

    def valid_step(self):
        """Validate for one step."""
        raise NotImplementedError

    def state_dict(self):
        """Return current states."""
        return {}

    def load_state_dict(self, sd):
        """Resume states."""
        raise NotImplementedError
