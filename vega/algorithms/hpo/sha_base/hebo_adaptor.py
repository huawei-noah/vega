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

"""Hebo adaptor."""

import logging
import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from vega.common.class_factory import ClassFactory, ClassType


logging.disable(logging.NOTSET)


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class HeboAdaptor(object):
    """An Hpo of BOHB."""

    def __init__(self, search_space=None, **kwargs):
        """Init BohbHpo."""
        space = self._to_hebo_serch_space(search_space)
        self.hebo = HEBO(space, model_name='gp')
        self.suggest_template = None
        self.search_space = search_space

    def _to_hebo_serch_space(self, search_space):
        space = []
        for hp in search_space["hyperparameters"]:
            if hp["type"] == "CATEGORY":
                hebo_hp = {
                    "name": hp["key"],
                    "type": "cat",
                    "categories": hp["range"]
                }
            elif hp["type"] in ["FLOAT", "FLOAT_EXP"]:
                hebo_hp = {
                    "name": hp["key"],
                    "type": "num",
                    "lb": hp["range"][0],
                    "ub": hp["range"][1],
                }
            elif hp["type"] in ["INT", "INT_EXP"]:
                hebo_hp = {
                    "name": hp["key"],
                    "type": "int",
                    "lb": hp["range"][0],
                    "ub": hp["range"][1],
                }
            elif hp["type"] == "BOOL":
                hebo_hp = {
                    "name": hp["key"],
                    "type": "bool",
                }
            else:
                raise Exception(f"HEBO does not support parameter type: {hp}")
            space.append(hebo_hp)
        space = DesignSpace().parse(space)
        return space

    def propose(self, num=1):
        """Propose a new sample."""
        suggestions = self.hebo.suggest(n_suggestions=num)
        recs = suggestions.to_dict()
        suggestions.drop(suggestions.index, inplace=True)
        self.suggest_template = suggestions
        out = []
        for index in list(recs.values())[0].keys():
            rec = {key: value[index] for key, value in recs.items()}
            rec = self.search_space.verify_constraints(rec)
            out.append(rec)
        return out

    def add(self, config, score):
        """Add a score."""
        rec = self.suggest_template.append(config, ignore_index=True)
        score = -1 * np.array([[score]])
        return self.hebo.observe(rec, score)
