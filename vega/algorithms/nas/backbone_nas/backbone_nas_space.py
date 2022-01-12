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

"""Check and Define Prune Model SearchSpace."""
import logging
from vega.common import ClassFactory, ClassType
from vega.core.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig
from vega.networks.network_desc import NetworkDesc
from vega.modules.operators import ops


@ClassFactory.register(ClassType.SEARCHSPACE)
class BackboneNasSearchSpace(SearchSpace):
    """BackboneNasSearchSpace."""

    @classmethod
    def get_space(self, desc):
        """Get model and input."""
        model = NetworkDesc(PipeStepConfig.model.model_desc).to_model()
        arch_params_key = 'network._arch_params.double_channels.{}.out_channels'
        search_space = [dict(key=arch_params_key.format(module.name), type="CATEGORY", range=[0, 1, 2])
                        for name, module in model.named_modules() if isinstance(module, ops.Conv2d)]
        logging.info("Backbone Nas Search Space: {}".format(search_space))
        return {"hyperparameters": search_space}
