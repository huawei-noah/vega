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
"""AutoGate Grda version Stage2 TrainerCallback."""

import logging
import pandas as pd
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.algorithms.nas.fis.ctr_trainer_callback import CtrTrainerCallback
from vega.core.pipeline.conf import ModelConfig

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoGateGrdaS2TrainerCallback(CtrTrainerCallback):
    """AutoGateGrdaS2TrainerCallback module."""

    def __init__(self):
        """Construct AutoGateGrdaS2TrainerCallback class."""
        super(CtrTrainerCallback, self).__init__()
        self.sieve_board = pd.DataFrame(
            columns=['selected_feature_pairs', 'score'])
        self.selected_pairs = list()

        logging.info("init autogate s2 trainer callback")

    def before_train(self, logs=None):
        """Call before_train of the managed callbacks."""
        super().before_train(logs)

        """Be called before the training process."""
        hpo_result = FileOps.load_pickle(FileOps.join_path(
            self.trainer.local_output_path, 'best_config.pickle'))
        logging.info("loading stage1_hpo_result \n{}".format(hpo_result))

        self.selected_pairs = hpo_result['feature_interaction']
        logging.info(f'feature_interaction: {self.selected_pairs}')

        # add selected_pairs
        setattr(ModelConfig.model_desc['custom'], 'selected_pairs', self.selected_pairs)

    def after_train(self, logs=None):
        """Call after_train of the managed callbacks."""
        curr_auc = float(self.trainer.valid_metrics.results['auc'])

        self.sieve_board = self.sieve_board.append(
            {
                'selected_feature_pairs': self.selected_pairs,
                'score': curr_auc
            }, ignore_index=True)
        result_file = FileOps.join_path(
            self.trainer.local_output_path, '{}_result.csv'.format(self.trainer.__worker_id__))

        self.sieve_board.to_csv(result_file, sep='\t')
