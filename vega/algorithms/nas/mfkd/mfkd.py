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

@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class MFKD1(SearchAlgorithm):
    def __init__(self, search_space):
        super(MFKD1, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space.search_space)
        self.max_samples = self.cfg["max_samples"]
        self.sample_count = 0
        self.acc_list = []

        self._get_all_arcs()
        self.points = list(np.random.choice(range(len(self.X)), size = self.max_samples, replace = False))
        logging.info('Selected %d random points: %s' % (len(self.points), str(self.points)))

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

    def _get_best_arc(self):
        """Find the best (by estimate) architecture from the search space"""

        X_train = []
        y_train = []

        for i in range(len(self.points)):
            idx = self.points[i]
            X_train.append(self.X[idx])
            y_train.append(self.acc_list[i])

        gpr = GPR(kernel = RBF(1.0))
        gpr.fit(X_train, y_train)

        preds = gpr.predict(self.X, return_std = True)
        best_idx = np.argmax(preds[0])

        return best_idx

    def search(self):
        idx = self.points[self.sample_count]
        logging.info('Checking architecture %d' % idx)
        desc = self._desc_from_choices(self.choices[idx])

        self.sample_count += 1
        self._save_model_desc_file(self.sample_count, desc)

        return self.sample_count, NetworkDesc(desc)

    def update(self, worker_path):

        with open(os.path.join(worker_path, 'performance.txt')) as infile:
            perf = infile.read()

        acc = eval(perf)[0]
        self.acc_list.append(acc)

        # not clear where to write the best architecture
        if self.is_completed:
            idx = self._get_best_arc()
            desc = self._desc_from_choices(self.choices[idx])
            logging.info('The best architecture %d, description %s' % (idx, str(desc)))

    @property
    def is_completed(self):
        """Check if the search is finished."""
        print(self.acc_list)
        return self.sample_count >= self.max_samples

    def _save_model_desc_file(self, id, desc):
        output_path = TaskOps(UserConfig().data.general).local_output_path
        desc_file = os.path.join(output_path, "nas", "model_desc_{}.json".format(id))
        FileOps.make_base_dir(desc_file)
        output = {}
        for key in desc:
            if key in ["type", "modules", "custom"]:
                output[key] = desc[key]
        with open(desc_file, "w") as f:
            json.dump(output, f)
