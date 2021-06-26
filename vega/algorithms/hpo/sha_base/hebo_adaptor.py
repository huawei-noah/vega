# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Hebo adaptor."""

import numpy as np
from vega.common.class_factory import ClassFactory, ClassType
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class HeboAdaptor(object):
    """An Hpo of BOHB."""

    def __init__(self, search_space=None, **kwargs):
        """Init BohbHpo."""
        space = self._to_hebo_serch_space(search_space)
        self.hebo = HEBO(space)
        self.suggest_template = None

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
            out.append(rec)
        return out

    def add(self, config, score):
        """Add a score."""
        rec = self.suggest_template.append(config, ignore_index=True)
        score = np.array([[score]])
        return self.hebo.observe(rec, score)
