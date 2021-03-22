# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Check and Define Prune Model SearchSpace."""
import logging
from zeus.common import ClassFactory, ClassType
from zeus.model_zoo import ModelZoo
from vega.core.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig


@ClassFactory.register(ClassType.SEARCHSPACE)
class PruneSearchSpace(SearchSpace):
    """Restrict and Terminate Base Calss."""

    @classmethod
    def get_space(self, desc):
        """Get model and input."""
        model_desc = PipeStepConfig.model.model_desc
        model = ModelZoo().get_model(dict(type='PruneDeformation', desc=model_desc))
        search_space = model.search_space
        params = []
        for key, value in search_space.items():
            hparam_name = 'network.props.{}'.format(key)
            params.append(dict(key=hparam_name, type="BINARY_CODE", range=[value]))
        params.append(dict(key='network.deformation', type="CATEGORY", range=['PruneDeformation']))
        logging.info("Prune Search Space: {}".format(params))
        return {"hyperparameters": params}
