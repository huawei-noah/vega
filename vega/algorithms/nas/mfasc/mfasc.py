# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Multi-fidelity Active Search with Co-kriging."""

import copy
from zeus.common import update_dict
from zeus.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
import itertools
from sklearn import preprocessing
import numpy as np
import logging

from . import mfasc_utils
from .conf import MFASCConfig

logger = logging.getLogger(__name__)
'''
Note: search steps must be performed successively
(parallel calls of the search method will violate the algorithms assumptions).
'''


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class MFASC(SearchAlgorithm):
    """Multi-fidelity Active Search with Co-kriging algorithm."""

    config = MFASCConfig()

    def __init__(self, search_space):
        """Construct the MFASC search class.

        :param search_space: config of the search space
        :type search_space: dictionary
        """
        super(MFASC, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space)
        self.budget_spent = 0
        self.batch_size = self.config.batch_size
        self.hf_epochs = self.config.hf_epochs
        self.lf_epochs = self.config.lf_epochs
        self.max_budget = self.config.max_budget  # total amount of epochs to train
        self.predictor = mfasc_utils.make_mf_predictor(self.config)
        self.r = self.config.fidelity_ratio  # fidelity ratio from the MFASC algorithm
        self.min_hf_sample_size = self.config.min_hf_sample_size
        self.min_lf_sample_size = self.config.min_lf_sample_size
        self.hf_sample = []  # pairs of (id, score)
        self.lf_sample = []  # pairs of (id, score)
        self.rho = self.config.prior_rho
        self.beta = self.config.beta
        self.cur_fidelity = None
        self.cur_i = None
        self.best_model_idx = None
        self._get_all_arcs()

    def search(self):
        """Search one random model.

        :return: total spent budget (training epochs), the model, and current training epochs (fidelity)
        :rtype: int, dict, int
        """
        remaining_hf_inds = np.array(list(set(range(len(self.X))) - set([x[0] for x in self.hf_sample])))
        remaining_lf_inds = np.array(list(set(range(len(self.X))) - set([x[0] for x in self.lf_sample])))
        if len(self.hf_sample) < self.min_hf_sample_size:
            # init random hf sample
            i = remaining_hf_inds[np.random.randint(len(remaining_hf_inds))]
            train_epochs = self.hf_epochs
            self.cur_fidelity = 'high'
        elif len(self.lf_sample) < self.min_lf_sample_size:
            # init random lf sample
            i = remaining_lf_inds[np.random.randint(len(remaining_lf_inds))]
            train_epochs = self.lf_epochs
            self.cur_fidelity = 'low'
        else:
            # update model
            X_low = np.array([self.X[s[0]] for s in self.lf_sample])
            y_low = np.array([s[1] for s in self.lf_sample])
            X_high = np.array([self.X[s[0]] for s in self.hf_sample])
            y_high = np.array([s[1] for s in self.hf_sample])
            self.predictor.fit(X_low, y_low, X_high, y_high)
            # main seach
            if (len(self.hf_sample) + len(self.lf_sample) + 1) % self.r == 0:
                # search hf
                inds = remaining_hf_inds[np.random.choice(len(remaining_hf_inds), self.batch_size, replace=False)]
                X_test = np.array([self.X[i] for i in inds])
                self.rho, mu, sigma = self.predictor.predict_hf(X_test)
                acquisition_score = mu + self.beta * sigma
                i = inds[np.argmax(acquisition_score)]
                self.cur_fidelity = 'high'
                train_epochs = self.hf_epochs
            else:
                # search lf
                inds = remaining_lf_inds[np.random.choice(len(remaining_lf_inds), self.batch_size, replace=False)]
                X_test = np.array([self.X[i] for i in inds])
                mu, sigma = self.predictor.predict_lf(X_test)
                if self.rho > 0:
                    acquisition_score = mu + self.beta * sigma
                else:
                    acquisition_score = mu - self.beta * sigma
                i = inds[np.argmin(acquisition_score)]
                train_epochs = self.lf_epochs
                self.cur_fidelity = 'low'

        desc = self._desc_from_choices(self.choices[i])
        self.budget_spent += train_epochs
        self.cur_i = i
        hps = {'trainer': {'epochs': train_epochs}}
        return self.budget_spent, desc, hps

    def update(self, report):
        """Update function.

        :param report: the serialized report.
        :type report: dict
        """
        logger.info(f'Updating, cur fidelity: {self.cur_fidelity}')
        acc = report['performance'].get('accuracy', np.nan)

        if self.cur_fidelity == 'high':
            self.hf_sample.append((self.cur_i, acc))

            self.best_model_idx = max(self.hf_sample, key=lambda x: x[1])[0]
            self.best_model_desc = self._desc_from_choices(self.choices[self.best_model_idx])
        elif self.cur_fidelity == 'low':
            self.lf_sample.append((self.cur_i, acc))
        else:
            raise ValueError(f'cur fidelity is {self.cur_fidelity}; it shall be either "high" or "low"')

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        :rtype: bool
        """
        return self.budget_spent > self.max_budget

    def _sub_config_choice(self, config, choices, pos):
        """Apply choices to config."""
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                _, pos = self._sub_config_choice(value, choices, pos)
            elif isinstance(value, list):
                choice = value[choices[pos]]
                config[key] = choice
                pos += 1

        return config, pos

    def _desc_from_choices(self, choices):
        """Create description object from choices."""
        desc = {}
        pos = 0

        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            module_cfg, pos = self._sub_config_choice(config_space, choices, pos)
            desc[key] = module_cfg

        desc = update_dict(desc, copy.deepcopy(self.search_space))

        return desc

    def _sub_config_all(self, config, vectors, choices):
        """Get all possible choices and their values."""
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                self._sub_config_all(value, vectors, choices)
            elif isinstance(value, list):
                vectors.append([float(x) for x in value])
                choices.append(list(range(len(value))))

    def _get_all_arcs(self):
        """Get all the architectures from the search space."""
        vectors = []
        choices = []

        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            self._sub_config_all(config_space, vectors, choices)

        self.X = list(itertools.product(*vectors))
        self.X = preprocessing.scale(self.X, axis=0)
        self.choices = list(itertools.product(*choices))

        logging.info('Number of architectures in the search space %d' % len(self.X))
