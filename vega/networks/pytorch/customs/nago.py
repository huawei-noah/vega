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

"""The NAGO model."""
import logging
import numpy as np
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory
from .utils.logical_graph import GeneratorSolution, LogicalMasterGraph
from .utils.layer import MasterNetwork


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.NETWORK)
class NAGO(Module):
    """Search space of NAGO."""

    def __init__(self, **kwargs):
        """Construct the Hierarchical Neural Architecture Generator class.

        :param net_desc: config of the searched structure
        """
        super(NAGO, self).__init__()
        logger.info("start init NAGO")
        # to prevent invalid graphs with G_nodes <= G_k
        kwargs['G1_K'] = int(np.min([kwargs['G1_nodes'] - 1, kwargs['G1_K']]))
        kwargs['G3_K'] = int(np.min([kwargs['G3_nodes'] - 1, kwargs['G3_K']]))
        logger.info("NAGO desc: {}".format(kwargs))

        top_graph_params = ['WS', kwargs['G1_nodes'], kwargs['G1_P'], kwargs['G1_K']]
        mid_graph_params = ['ER', kwargs['G2_nodes'], kwargs['G2_P']]
        bottom_graph_params = ['WS', kwargs['G3_nodes'], kwargs['G3_P'], kwargs['G3_K']]

        channel_ratios = [kwargs['ch1_ratio'], kwargs['ch2_ratio'], kwargs['ch3_ratio']]
        stage_ratios = [kwargs['stage1_ratio'], kwargs['stage2_ratio'], kwargs['stage3_ratio']]

        conv_type = 'normal'
        top_merge_dist = [1.0, 0.0, 0.0]
        mid_merge_dist = [1.0, 0.0, 0.0]
        bottom_merge_dist = [1.0, 0.0, 0.0]
        op_dist = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        solution = GeneratorSolution(top_graph_params, mid_graph_params, bottom_graph_params,
                                     stage_ratios, channel_ratios, op_dist, conv_type,
                                     top_merge_dist, mid_merge_dist, bottom_merge_dist)

        # Generate an architecture from the generator
        model_frame = LogicalMasterGraph(solution)
        # Compute the channel multipler factor based on the parameter count limit
        n_params_base = model_frame._get_param_count()
        multiplier = int(np.sqrt(float(kwargs['n_param_limit']) / n_params_base))
        self.model = MasterNetwork(model_frame, multiplier, kwargs['image_size'],
                                   kwargs['num_classes'], None, False)

    def forward(self, x):
        """Calculate the output of the model.

        :param x: input tensor
        :return: output tensor of the model
        """
        y, aux_logits = self.model(x)
        return y
