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

"""
Asynchronous Successive Halving Algorithm.

https://arxiv.org/abs/1810.05934

https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/
.. code-block:: python
    Detail of ASHA:
    Input: minimum resource r, maximum resource R, reduction factor η, minimum
        early-stopping rate s
    Algorithm: ASHA()
        repeat
            for each free worker do
                (θ; k) = get_job()
                run_then_return_val_loss(θ; rη^(s+k))
                end
            for completed job (θ, k) with loss l do
                Update configuration θ in rung k with loss l.
            end
    Procedure: get_job()
        // Check to see if there is a promotable config.
        for k = |log_η(R=r)| − s; ... ; 1; 0 do
            candidates = top_k(rung k; |rung k|=η)
            promotable = ft for t 2 candidates if t not already promotedg
            if |promotable| > 0 then
                return promotable[0]; k + 1
            end
        end
        Draw random configuration θ. // If not, grow bottom rung.
        return θ; 0
"""
import math
import logging
import random
from math import log
import numpy as np
from vega.common.pareto_front import get_pareto
from .sha_base import ShaBase
from .status_type import StatusType


logger = logging.getLogger(__name__)


class ASHA(ShaBase):
    """ASHA (Asynchronous Successive Halving Algorithm).

    :param search_space: a pre-defined search space.
    :type search_space: object, instance os `SearchSpace`.
    :param int num_samples: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param min_epochs: `min_epochs` is the init min epoch.
    :type min_epochs: int, default is 1.
    :param eta: rung base `eta`.
    :type eta: int, default is 3.
    """

    def __init__(self, search_space, num_samples, max_epochs, min_epochs=1,
                 eta=3, random_config=True, start_config_id=0):
        """Init ASHA."""
        super().__init__(search_space, num_samples, max_epochs, min_epochs, eta)
        self.s_max = int(log(max_epochs / min_epochs) / log(eta))
        self.single_epoch = min_epochs
        # minimum early-stopping rate s
        self.sr = 0
        self.total_rungs = self.s_max + 1 - self.sr
        self.eta = eta
        # init all the configs
        config_list = self.get_hyperparameters(num_samples)
        for i, config in enumerate(config_list):
            if random_config:
                self.all_config_dict[i] = config
            tmp_row_data = {'rung_id': 0,
                            'config_id': i + start_config_id,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
        logger.info(f"ashs info, total rungs: {self.total_rungs}, "
                    f"min_epochs: {self.single_epoch}, eta: {self.eta}")

    def add_score(self, config_id, rung_id, score):
        """Update the sieve_board for add score.

        :param int config_id: Description of parameter `config_id`.
        :param int rung_id: Description of parameter `rung_id`.
        :param float score: Description of parameter `score`.

        """
        loc = self.sieve_board.loc[(self.sieve_board['config_id'] == config_id) & (
            self.sieve_board['rung_id'] == rung_id), ['status', 'score']]
        logger.debug(f"update sieve board loc:\n{loc}")
        self.sieve_board.loc[(self.sieve_board['config_id'] == config_id) & (
            self.sieve_board['rung_id'] == rung_id), ['status', 'score']] = [StatusType.FINISHED, score]
        self.is_completed = self._check_completed()
        logger.info(f"add score\n{self.sieve_board}")
        return

    def propose(self):
        """Propose next hyper parameter.

        :return: dict of a proposed config.
        :rtype: dict, {'config_id': int,
                       'rung_id': int,
                       'configs': dict,
                       'epoch': int}

        """
        # Check to see if there is a promotable config
        for rung_id in reversed(range(0, self.total_rungs - 1)):
            candidate_ids = self._get_top_k_config_ids(rung_id)
            if candidate_ids is not None:
                promote_rung_id = rung_id + 1
                s_epoch = self.single_epoch * math.pow(
                    self.eta, promote_rung_id + self.sr)
                promote_config_id = candidate_ids[0]
                results = {
                    'config_id': promote_config_id,
                    'rung_id': promote_rung_id,
                    'configs': self.get_config(promote_config_id),
                    'epoch': int(s_epoch)
                }
                self._change_status(rung_id=rung_id,
                                    config_id=promote_config_id,
                                    status=StatusType.PORMOTED)
                self._change_status(rung_id=promote_rung_id,
                                    config_id=promote_config_id,
                                    status=StatusType.RUNNING)
                self.total_propose = self.total_propose + 1
                logger.info(f"propose new config, asha\n{self.sieve_board}")
                return results

        # Draw random configuration θ from bottom rung.
        bottom_rung = 0
        _key = (self.sieve_board['rung_id'] == bottom_rung) & \
               (self.sieve_board['status'] == StatusType.WAITTING)
        rung_df = self.sieve_board.loc[_key]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        results = {
            'config_id': next_config_id,
            'rung_id': bottom_rung,
            'configs': self.get_config(next_config_id),
            'epoch': int(self.single_epoch)
        }
        self._change_status(rung_id=bottom_rung,
                            config_id=next_config_id,
                            status=StatusType.RUNNING)
        self.total_propose = self.total_propose + 1
        logger.info(f"propose new config, asha\n{self.sieve_board}")
        return results

    def get_config(self, config_id):
        """Get config."""
        return self.all_config_dict[config_id]

    def next_rung(self, config_id):
        """Prompt next rung."""
        for rung_id in reversed(range(0, self.total_rungs - 1)):
            candidate_ids = self._get_top_k_config_ids(rung_id)
            if candidate_ids is not None:
                if config_id not in candidate_ids:
                    continue
                promote_rung_id = rung_id + 1
                s_epoch = self.single_epoch * math.pow(
                    self.eta, promote_rung_id + self.sr)
                promote_config_id = config_id
                results = {
                    'config_id': promote_config_id,
                    'rung_id': promote_rung_id,
                    'configs': self.get_config(promote_config_id),
                    'epoch': int(s_epoch)
                }
                self._change_status(rung_id=rung_id,
                                    config_id=promote_config_id,
                                    status=StatusType.PORMOTED)
                self._change_status(rung_id=promote_rung_id,
                                    config_id=promote_config_id,
                                    status=StatusType.RUNNING)
                self.total_propose = self.total_propose + 1
                logger.info(f"pormoted existed config\n{self.sieve_board}")
                return results
        return None

    def _add_to_board(self, one_dict):
        """Add the new record into board.

        :param dict one_dict: Description of parameter `one_dict`.

        """
        # TODO pandas use contact replace append function
        self.sieve_board = self.sieve_board.append(one_dict, ignore_index=True)

    def _change_status(self, rung_id, config_id, status):
        """Change the status of each config.

        :param int rung_id: Description of parameter `rung_id`.
        :param int config_id: Description of parameter `config_id`.
        :param type status: Description of parameter `status`.
        :type enum: StatusType

        """
        _key = (self.sieve_board['config_id'] == config_id) & \
               (self.sieve_board['rung_id'] == rung_id)
        change_df = self.sieve_board.loc[_key]
        if change_df.empty:
            tmp_row_data = {'rung_id': rung_id,
                            'config_id': config_id,
                            'status': status}
            self._add_to_board(tmp_row_data)
        else:
            self.sieve_board.loc[_key, ['status']] = [status]

    def _check_completed(self):
        """Check task is completed.

        :return: if the search algorithm is finished.
        :rtype: bool.

        """
        current_rung_df = self.sieve_board.loc[
            self.sieve_board['status'].isin([StatusType.WAITTING, StatusType.RUNNING])]
        if not current_rung_df.empty:
            return False

        max_rung_id = self.sieve_board['rung_id'].max()
        if max_rung_id >= self.total_rungs - 1:
            return True

        candidate_ids = self._get_top_k_config_ids(max_rung_id)
        if candidate_ids is not None:
            return False
        return True

    def _get_top_k_config_ids(self, rung_id):
        """Get top k configs.

        :param int rung_id: Description of parameter `rung_id`.
        :return: a list of top k config id.
        :rtype: list or None.

        """
        board = self.sieve_board
        _key = (board['rung_id'] == rung_id) & \
               (board['status'].isin([StatusType.FINISHED, StatusType.PORMOTED]))
        df = board.loc[_key]
        if df.empty:
            return None
        else:
            k = int(len(df.index) / self.eta)
            if k <= 0:
                return None

            num_next_rung = board[(board['rung_id'] == rung_id + 1)].shape[0]
            if num_next_rung >= k:
                return None

            if isinstance(df.iloc[0]["score"], float) or isinstance(df.iloc[0]["score"], int):
                ids = df.sort_values("score", ascending=False).iloc[:k]["config_id"].tolist()
            elif isinstance(df.iloc[0]["score"], list):
                data = df[["config_id", "score"]].to_numpy()
                id_score = np.hstack((data[:, 0].reshape(data[:, 0].shape[0], 1), np.vstack(data[:, 1])))

                # remove same score cols
                cols = [i for i in range(1, id_score.shape[1]) if (id_score[:, i] == id_score[0, i]).all()]
                id_score = np.delete(id_score, cols, axis=1)

                if id_score.shape[1] == 2:
                    id_score = id_score.tolist()
                    id_score = sorted(id_score, key=(lambda x: x[1]), reverse=True)
                    ids = [x[0] for x in id_score[:k]]
                else:
                    pareto = get_pareto(id_score, index=True)[:, 0].T.tolist()
                    if len(pareto) > k:
                        ids = random.sample(pareto, k)
                    elif len(pareto) < k:
                        others = data[:, 0].tolist()
                        others = [item for item in others if item not in pareto]
                        others = random.sample(others, k - len(pareto))
                        ids = pareto + others
                    else:
                        ids = pareto
            else:
                logger.error(f"invalid score: {df.iloc[0]['score']}")
            candidate_df = df.loc[(df['config_id'].isin(ids)) & (df["status"] == StatusType.FINISHED)]
            if candidate_df.empty:
                return None
            else:
                return candidate_df['config_id'].tolist()
