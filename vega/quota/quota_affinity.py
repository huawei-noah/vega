# -*- coding:utf-8 -*-

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

"""Quota for Affinity."""

import logging
import ast
import pandas as pd
from sklearn import ensemble


class AffinityModelBase(object):
    """Base class for affinity regression."""

    def __init__(self, affinity_file, affinity_value):
        self.affinity_report = pd.read_csv(affinity_file)
        self.standard = affinity_value
        self.ml_model = ensemble.RandomForestClassifier(n_estimators=20)

    def build_model(self):
        """Build regression model."""
        desc = self.affinity_report['desc']
        inputs = self.generate_input_space(desc)
        labels = self.generate_label()
        self.ml_model.fit(inputs, labels)

    def predict(self, input):
        """Predict output from input."""
        return self.ml_model.predict(input[:1])[0]

    def generate_input_space(self, desc):
        """Generate input space from desc."""
        if desc is None:
            return None
        space_list = []
        for idx in range(len(desc)):
            desc_item = ast.literal_eval(desc.iloc[idx])
            space_dict = {}
            self.init_space_dict(space_dict)
            for key, value in desc_item.items():
                self.get_space_dict(key, value, space_dict)
            if space_dict:
                space_list.append(space_dict)
        return pd.DataFrame(space_list)

    def generate_label(self):
        """Generate label from affinity report."""
        _pfms = self.affinity_report['performance']
        _metric_key = ast.literal_eval(self.affinity_report['_objective_keys'][0])[0]
        label_list = []
        for pfm in _pfms:
            value = ast.literal_eval(pfm)[_metric_key]
            clc = 1 if value > self.standard else 0
            label_list.append({_metric_key: clc})
        return pd.DataFrame(label_list)

    def init_space_dict(self, space_dict):
        """Initialize space dict."""
        pass

    def get_space_dict(self, *args):
        """Get space dict from desc."""
        raise NotImplementedError


class AffinityModelSrea(AffinityModelBase):
    """Affinity Regression for SR-EA."""

    def __init__(self, affinity_file, affinity_value):
        super(AffinityModelSrea, self).__init__(affinity_file, affinity_value)

    def init_space_dict(self, space_dict):
        """Initialize space dict."""
        for i in range(80):
            space_dict['code_{}'.format(i)] = False

    def get_space_dict(self, key, value, space_dict):
        """Get space dict from desc."""
        key = key.split('.')[-1]
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.get_space_dict(sub_key, sub_value, space_dict)
        elif key == 'code' and isinstance(value, str):
            for i, element in enumerate(value):
                if element == '0':
                    space_dict[key + '_{}'.format(i)] = 1
                elif element == '1':
                    space_dict[key + '_{}'.format(i)] = 2
                else:
                    space_dict[key + '_{}'.format(i)] = 3


class QuotaAffinity(object):
    """Generate affinity model of search space, filter bad sample."""

    def __init__(self, affinity_cfg):
        affinity_class = ast.literal_eval(self.get_affinity_model(affinity_cfg.type))
        self.affinity_model = affinity_class(affinity_cfg.affinity_file, affinity_cfg.affinity_value)
        self.affinity_model.build_model()

    def get_affinity_model(self, affinity_type):
        """Get specific affinity model name."""
        affinity_model_dict = {
            'sr_ea': 'AffinityModelSrea'
        }
        return affinity_model_dict[affinity_type]

    def is_affinity(self, desc):
        """Judge the desc is affinity or not."""
        desc_dict = {'desc': str(desc)}
        input = pd.DataFrame([desc_dict])
        input = self.affinity_model.generate_input_space(input['desc'])
        try:
            result = self.affinity_model.predict(input)
        except Exception:
            logging.info('The sampled desc is not affinity')
            return False
        if result == 1:
            logging.info('The sampled desc is affinity')
            return True
        else:
            logging.info('The sampled desc is not affinity')
            return False
