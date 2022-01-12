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
"""Implementation of Ada-Segment algorithm."""
from enum import Enum
from torch import nn
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam


class WeightController(nn.Module):
    """WeightController network of ada-segment."""

    def __init__(self, in_features=10, hidden_size=16):
        super(WeightController, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, in_features),
        )
        for m in self.mlp.children():
            if isinstance(m, nn.Linear):
                mean = 1 / m.in_features
                std = mean / 3
                nn.init.normal(m.weight, mean, std)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Calculate the output of the network."""
        return self.mlp(x)


class Xloss(nn.Module):
    """Loss function of weight controller."""

    def __init__(self, total_rungs):
        super(Xloss, self).__init__()
        self.T = total_rungs
        self.eps = 1e-8

    def forward(self, cur_performance, last_best, cur_rung, output):
        """Calculate the loss."""
        cur_performance = np.array(cur_performance)
        reward_local = (cur_performance - np.mean(cur_performance)) / (np.std(cur_performance) + self.eps)
        reward_improve = (cur_performance - last_best) / (np.std(cur_performance - last_best) + self.eps)
        reward_all = cur_rung / self.T * (reward_local + reward_improve)

        loss = torch.tensor([0])
        for i in range(reward_all.size):
            loss = loss + torch.Tensor([reward_all[i]]) * output
        loss = torch.mean(loss)
        return loss


class StatusType(Enum):
    """StatusType."""

    KILLED = 1
    RUNNING = 2
    WAITTING = 3
    FINISHED = 4


class AdaSegment(object):
    """Implementation of AdaSegment."""

    def __init__(self, model_num, total_rungs, loss_num, best_loss=None):
        self.model_num = model_num
        self.total_rungs = total_rungs
        self.loss_num = loss_num
        self.best_loss = best_loss
        self.sieve_board = pd.DataFrame(columns=['rung_id', 'config_id', 'status', 'score', 'loss'])
        self.is_completed = False
        self.rung_id = 0
        self.estimator = WeightController(in_features=loss_num)
        self.optimizer = Adam(self.estimator.parameters(), lr=5e-2, weight_decay=5e-4)
        self.loss = Xloss(total_rungs=total_rungs)

        for i in range(model_num):
            init_info = {'rung_id': 0, 'config_id': i, 'status': StatusType.WAITTING, 'score': 0,
                         'loss': [1] * self.loss_num}
            self.sieve_board = self.sieve_board.append(init_info, ignore_index=True)

    def propose(self):
        """Propose a sample."""
        _key = (self.sieve_board['rung_id'] == self.rung_id) & \
               (self.sieve_board['status'] == StatusType.WAITTING)
        rung_df = self.sieve_board.loc[_key]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        _key = (self.sieve_board['rung_id'] == self.rung_id) & (self.sieve_board["config_id"] == next_config_id)
        self.sieve_board.loc[_key, 'status'] = StatusType.RUNNING
        sample = {"config_id": next_config_id,
                  "rung_id": self.rung_id,
                  "dynamic_weight": None}
        if self.best_loss is None:
            sample["dynamic_weight"] = [1] * self.loss_num
        else:
            input_tensor = torch.Tensor([self.best_loss])
            weighted_loss = self.estimator(input_tensor)
            self.weight_tensor = weighted_loss / input_tensor
            self.weight_tensor = self.weight_tensor.squeeze()
            weight = self.weight_tensor.detach().numpy().tolist()
            new_weight = np.random.normal(weight, 0.2)
            sample["dynamic_weight"] = new_weight
        return sample

    def add_score(self, config_id, rung_id, reward, cur_loss):
        """Update the sieve_board.

        :param config_id: the id number of the config
        :type config_id: int
        :param rung_id: the id number of the rung
        :type rung_id: int
        :param reward: the evaluate result of the config
        :type: float
        :param cur_loss: the vector of the loss
        :type cur_loss: list

        """
        _key = (self.sieve_board['rung_id'] == rung_id) & (self.sieve_board['config_id'] == config_id)
        self.sieve_board.loc[_key, 'status'] = StatusType.FINISHED
        self.sieve_board.loc[_key, 'score'] = reward
        row_index = rung_id * self.model_num + config_id
        self.sieve_board['loss'] = self.sieve_board['loss'].astype('object')
        self.sieve_board.at[row_index, 'loss'] = cur_loss
        if self._check_rung_completed():
            self._init_next_rung()
        self.is_completed = self._check_completed()

    def _init_next_rung(self):
        current_rung_df = self.sieve_board.loc[(self.sieve_board['rung_id'] == self.rung_id)]
        if self.rung_id >= 1:
            last_rung_df = self.sieve_board.loc[(self.sieve_board['rung_id'] == self.rung_id - 1)]
            cur_performance = current_rung_df['score'].to_list()
            last_best = max(last_rung_df['score'].to_list())
            loss = self.loss(cur_performance=cur_performance, last_best=last_best, cur_rung=self.rung_id + 1,
                             output=self.weight_tensor)
            loss.backward()
            self.optimizer.step()

        current_rung_df.sort_values(by="score", axis=0, ascending=False, inplace=True)
        self.best_score = current_rung_df.iloc[0]["score"]
        self.best_loss = current_rung_df.iloc[0]["loss"]

        self.rung_id += 1
        for i in range(self.model_num):
            init_info = {'rung_id': self.rung_id, 'config_id': i, 'status': StatusType.WAITTING}
            self.sieve_board = self.sieve_board.append(init_info, ignore_index=True)

    def _check_completed(self):
        all_rung_df = self.sieve_board.loc[self.sieve_board['status'].isin(
            [StatusType.WAITTING, StatusType.RUNNING])
        ]
        if all_rung_df.empty and self.rung_id >= self.total_rungs:
            return True
        else:
            return False

    def _check_rung_completed(self):
        current_rung_df = self.sieve_board.loc[(self.sieve_board['rung_id'] == self.rung_id) & (
            self.sieve_board['status'].isin([StatusType.WAITTING, StatusType.RUNNING]))]
        if current_rung_df.empty:
            return True
        else:
            return False
