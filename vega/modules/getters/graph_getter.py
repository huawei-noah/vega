# -*- coding: utf-8 -*-

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

"""Getter for Graph."""
from collections import OrderedDict
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module
from vega.model_zoo import ModelZoo
from vega.modules.graph_utils import graph2desc
from vega import is_tf_backend


@ClassFactory.register(ClassType.NETWORK)
class GraphGetter(Module):
    """Get output layer by layer names and connect into a OrderDict."""

    def __init__(self, desc=None, weight_file=None, pb_file=None):
        super(GraphGetter, self).__init__()
        if isinstance(desc, dict):
            src_model = ModelZoo().get_model(desc)
        else:
            src_model = desc
        weights = OrderedDict()
        if is_tf_backend():
            import tensorflow.compat.v1 as tf
            from tensorflow.python.framework import tensor_util
            tf.reset_default_graph()
            data_shape = (1, 224, 224, 3)
            x = tf.ones(data_shape)
            if pb_file:
                with tf.io.gfile.GFile(pb_file, 'rb') as f:
                    graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                graph = tf.Graph()
                with graph.as_default():
                    tf.import_graph_def(graph_def, name='')
                weight_file = None
                wts = [n for n in graph_def.node if n.op == 'Const']
                for n in wts:
                    weights[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)
            else:
                src_model(x, self.training)
                graph = tf.get_default_graph()
            desc = graph2desc(graph)
            tf.reset_default_graph()
        self.model = ModelZoo().get_model(desc, weight_file)
        if weights:
            self.model.load_checkpoint_from_numpy(weights)
