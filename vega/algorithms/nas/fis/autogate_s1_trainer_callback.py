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

"""AutoGate top-k version Stage1 TrainerCallback."""

import logging
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.algorithms.nas.fis.ctr_trainer_callback import CtrTrainerCallback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoGateS1TrainerCallback(CtrTrainerCallback):
    """AutoGateS1TrainerCallback module."""

    def __init__(self):
        """Construct AutoGateS1TrainerCallback class."""
        super(CtrTrainerCallback, self).__init__()
        self.best_score = 0

        logging.info("init autogate s1 trainer callback")

    def after_valid(self, logs=None):
        """Call after_valid of the managed callbacks."""
        self.model = self.trainer.model
        feature_interaction_score = self.model.get_feature_interaction_score()
        print('get feature_interaction_score', feature_interaction_score)

        curr_auc = float(self.trainer.valid_metrics.results['auc'])
        if curr_auc > self.best_score:
            best_config = {
                'score': curr_auc,
                'feature_interaction_score': feature_interaction_score
            }

            logging.info("BEST CONFIG IS\n{}".format(best_config))
            pickle_result_file = FileOps.join_path(
                self.trainer.local_output_path, 'best_config.pickle')
            logging.info("Saved to {}".format(pickle_result_file))
            FileOps.dump_pickle(best_config, pickle_result_file)

            self.best_score = curr_auc
