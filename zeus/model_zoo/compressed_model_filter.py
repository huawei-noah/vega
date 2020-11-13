# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Compressed model filter."""
import os
import pandas as pd


class CompressedModelFilter(object):
    """Compressed Model Filter."""

    def __init__(self, models_file):
        """Initialize."""
        self.models_file = models_file
        self.model_zoo = self._parse_file(self.models_file)

    def _parse_file(self, models_file):
        """Parse model files by pandas."""
        model_zoo = pd.read_csv(models_file)
        return model_zoo

    def _parse_standard(self, standard):
        """Parse quota standard to target and restrict."""
        restrict = standard.restrict().to_json()
        target = standard.target().to_json()
        return target, restrict

    def _filtrate(self, restrict):
        """Filtrate models by restrict condition."""
        filters = []
        condition = True
        for key, value in restrict.items():
            if not value or key not in self.model_zoo:
                continue
            filters.append(self.model_zoo[key].map(lambda x: x < value))
            condition = condition & filters[-1]
        filtered_data = self.model_zoo[condition]
        return filtered_data

    def _choose_models(self, candidates, target, num):
        """Choose models by target type and selected number."""
        sort_cands = candidates.sort_values(target.type, inplace=False)
        # sort_cands = sort_cands[sort_cands[target.type] > target.value]
        if len(sort_cands) < num:
            num = len(sort_cands)
        satisfied_models = []
        for idx in range(num):
            satisfied_models.append(sort_cands.iloc[idx]['desc'])
        return satisfied_models

    def select_satisfied_model(self, standard, num):
        """Select satisfied models by standard."""
        target, restrict = self._parse_standard(standard)
        candidates = self._filtrate(restrict)
        satisfied_models = self._choose_models(candidates, target, num)
        return satisfied_models
