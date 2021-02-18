# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Create tf utils for assign weights between learner and actor and model utils for universal usage."""
import os
import numpy as np
from collections import OrderedDict, deque
from absl import logging

import tensorflow as tf


class TFVariables:
    """Set & Get weights for TF networks with actor's route."""

    def __init__(self, output_op, session):
        """Extract variables, makeup the TFVariables class."""
        self.session = session
        if not isinstance(output_op, (list, tuple)):
            output_op = [output_op]

        track_explored_ops = set(output_op)
        to_process_queue = deque(output_op)
        to_handle_node_list = list()

        # find the dependency variables start with inputs with BFS.
        while len(to_process_queue) != 0:
            tf_object = to_process_queue.popleft()
            if tf_object is None:
                continue

            if hasattr(tf_object, "op"):
                tf_object = tf_object.op
            for input_op in tf_object.inputs:
                if input_op not in track_explored_ops:
                    to_process_queue.append(input_op)
                    track_explored_ops.add(input_op)

            # keep track of explored operations,
            for control in tf_object.control_inputs:
                if control not in track_explored_ops:
                    to_process_queue.append(control)
                    track_explored_ops.add(control)

            # process the op with 'Variable' or 'VarHandle' attribute
            if "VarHandle" in tf_object.node_def.op or "Variable" in tf_object.node_def.op:
                to_handle_node_list.append(tf_object.node_def.name)

        self.node_hub_with_order = OrderedDict()
        # go through whole global variables
        for _val in tf.global_variables():
            if _val.op.node_def.name in to_handle_node_list:
                self.node_hub_with_order[_val.op.node_def.name] = _val

        self._ph, self._to_assign_node_dict = dict(), dict()

        for node_name, variable in self.node_hub_with_order.items():
            self._ph[node_name] = tf.placeholder(variable.value().dtype,
                                                 variable.get_shape().as_list(),
                                                 name="ph_{}".format(node_name))
            self._to_assign_node_dict[node_name] = variable.assign(self._ph[node_name])

        logging.debug("node_hub_with_order: \n{}".format(self.node_hub_with_order.keys()))

    def get_weights(self):
        """Get weights with dict type."""
        _weights = self.session.run(self.node_hub_with_order)
        return _weights

    def set_weights(self, to_weights):
        """Set weights with dict type."""
        nodes_to_assign = [
            self._to_assign_node_dict[node_name] for node_name in to_weights.keys()
            if node_name in self._to_assign_node_dict
        ]
        # unused_nodes = [_node for _node in to_weights.keys()
        #                 if _node not in self._to_assign_node_dict]
        # assert not unused_nodes, "weights: {} not assign!".format(unused_nodes)

        if not nodes_to_assign:
            raise KeyError("NO node's weights could assign in self.graph")

        assign_feed_dict = {
            self._ph[node_name]: value
            for (node_name, value) in to_weights.items() if node_name in self._ph
        }

        self.session.run(
            nodes_to_assign,
            feed_dict=assign_feed_dict,
        )

    def save_weights(self, save_name: str):
        """Save weights with numpy io."""
        _weights = self.session.run(self.node_hub_with_order)
        np.savez(save_name, **_weights)

    @staticmethod
    def read_weights(weight_file: str):
        """Read weights with numpy.npz."""
        np_file = np.load(weight_file)
        return OrderedDict(**np_file)

    def set_weights_with_npz(self, npz_file: str):
        """Set weight with numpy file."""
        weights = self.read_weights(npz_file)
        self.set_weights(weights)
