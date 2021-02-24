import os
import copy
import json
import random
import torch.nn as nn
import vega
from vega.core.common.utils import update_dict
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import UserConfig, TaskOps, FileOps
from vega.search_space.networks import NetTypes, NetworkFactory, NetworkDesc
from vega.search_space.search_algs import SearchAlgorithm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import itertools
from sklearn import preprocessing
import numpy as np
import logging

import vega.algorithms.nas.mfasc.mfasc_utils as mfasc_utils

logger = logging.getLogger(__name__)
'''
Note: search steps must be performed successively 
(parallel calls of the search method will violate the algorithms assumptions).
'''

# todo: remove hard-coded constants
@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class MFASC(SearchAlgorithm):

    def __init__(self, search_space):
        super(MFASC, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space.search_space)
        self.budget_spent = 0
        self.batch_size = 1000

        #seed = self.cfg.get('seed', 99999)
        #np.random.seed(seed)

        self.hf_epochs = self.cfg['hf_epochs'] 
        self.lf_epochs = self.cfg['lf_epochs']
        self.max_budget = self.cfg['max_budget'] # total amount of epochs to train
        self.predictor = mfasc_utils.make_mf_predictor()
        self.r = self.cfg['fidelity_ratio']  # fidelity ratio from the MFASC algorithm
        self.min_hf_sample_size = self.cfg['min_hf_sample_size'] 
        self.min_lf_sample_size = self.cfg['min_lf_sample_size'] 
        self.hf_sample = [] # pairs of (id, score)
        self.lf_sample = [] # pairs of (id, score)
        self.rho = 1.0
        self.beta = 1.0
        self.cur_fidelity = None
        self.cur_i = None

        self.best_model_idx = None

        self._get_all_arcs()

    def search(self):
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
        return self.budget_spent, NetworkDesc(desc), train_epochs 

    def update(self, worker_path):
        logger.info(f'Updating, cur fidelity: {self.cur_fidelity}')

        with open(os.path.join(worker_path, 'performance.txt')) as infile:
            perf = infile.read()

        acc = eval(perf)[0]

        if self.cur_fidelity == 'high':
            self.hf_sample.append((self.cur_i, acc))

            self.best_model_idx = max(self.hf_sample, key = lambda x : x[1])[0]
            self.best_model_desc = self._desc_from_choices(self.choices[self.best_model_idx])
        else:
            self.lf_sample.append((self.cur_i, acc))

    @property
    def is_completed(self):
        return self.budget_spent > self.max_budget


    def _sub_config_choice(self, config, choices, pos):
        """Apply choices to config"""

        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                _, pos = self._sub_config_choice(value, choices, pos)
            elif isinstance(value, list):
                choice = value[choices[pos]]
                config[key] = choice
                pos += 1

        return config, pos

    def _desc_from_choices(self, choices):
        """Create description object from choices"""

        desc = {}
        pos = 0

        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            module_cfg, pos = self._sub_config_choice(config_space, choices, pos)
            desc[key] = module_cfg

        desc = update_dict(desc, copy.deepcopy(self.search_space))

        return desc

    def _sub_config_all(self, config, vectors, choices):
        """Get all possible choices and their values"""

        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                self._sub_config_all(value, vectors, choices)
            elif isinstance(value, list):
                vectors.append([float(x) for x in value])
                choices.append(list(range(len(value))))

    def _get_all_arcs(self):
        """Get all the architectures from the search space"""

        vectors = []
        choices = []

        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            self._sub_config_all(config_space, vectors, choices)

        self.X = list(itertools.product(*vectors))
        self.X = preprocessing.scale(self.X, axis = 0)
        self.choices = list(itertools.product(*choices))

        logging.info('Number of architectures in the search space %d' % len(self.X))
