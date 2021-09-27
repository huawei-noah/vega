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
from vega.common import ClassFactory, ClassType
from vega.core.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig
from vega.networks.network_desc import NetworkDesc
from vega.modules.operators import ops


def is_depth_wise_conv(module):
    """Determine Conv2d."""
    if hasattr(module, "groups"):
        return module.groups != 1 and module.in_channels == module.out_channels
    elif hasattr(module, "group"):
        return module.group != 1 and module.in_channels == module.out_channels


@ClassFactory.register(ClassType.SEARCHSPACE)
class PruneSearchSpace(SearchSpace):
    """Prune SearchSpace."""

    @classmethod
    def get_space(self, desc):
        """Get model and input."""
        model = NetworkDesc(PipeStepConfig.model.model_desc).to_model()
        arch_params_key = 'network._arch_params.Prune.{}.out_channels'
        search_space = [dict(key=arch_params_key.format(module.name), type="HALF", range=[module.out_channels])
                        for name, module in model.named_modules() if
                        isinstance(module, ops.Conv2d) and not is_depth_wise_conv(module)]
        logging.info("Prune Nas Search Space: {}".format(search_space))
        return {"hyperparameters": search_space}
