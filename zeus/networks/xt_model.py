# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""xingtian model."""
import tensorflow as tf
from tensorflow.python.keras.losses import MSE

from zeus.common import ClassFactory, ClassType
from zeus.modules.module import Module
from zeus.modules.operators.ops import Relu, Linear, Conv2d, View, Lambda


@ClassFactory.register(ClassType.NETWORK)
class DqnMlpNet(Module):
    """Create DQN Mlp net with FineGrainedSpace."""

    def __init__(self, **descript):
        """Create layers."""
        super().__init__()
        state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")
        hidden_size = descript.get("hidden_size", 128)

        self.fc1 = Linear(state_dim, hidden_size)
        self.ac1 = Relu()
        self.fc2 = Linear(hidden_size, action_dim)


@ClassFactory.register(ClassType.NETWORK)
class DqnCnnNet(Module):
    """Create DQN Cnn net with FineGrainedSpace."""

    def __init__(self, **descript):
        """Create layers."""
        super().__init__()
        # state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")

        self.lambda1 = Lambda(lambda x: tf.cast(x, dtype='float32') / 255.)
        self.conv1 = Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, bias=False)
        self.ac1 = Relu()
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.ac2 = Relu()
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False)
        self.ac3 = Relu()
        self.view = View()
        self.fc1 = Linear(64, 256)
        self.ac4 = Relu()
        self.fc2 = Linear(256, action_dim)


@ClassFactory.register(ClassType.LOSS, 'mse_loss')
def mse_loss(logits, labels):
    """Mse loss."""
    return tf.reduce_mean(MSE(logits, labels))
