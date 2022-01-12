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
"""Convert pytorch weight to tf checkpoint."""
import logging
import re
from collections import OrderedDict
import numpy


def get_assignment_map(checkpoint_path, pop_global_step=True):
    """Get assignment from checkpoint path."""
    import tensorflow.compat.v1 as tf
    assignment_map = OrderedDict()
    for name, var in tf.train.list_variables(checkpoint_path):
        if pop_global_step and tf.GraphKeys.GLOBAL_STEP == name:
            continue
        if not name.endswith('/') and name.split('/'):
            name = name[:name.index(name.split('/')[-1])]
        assignment_map[name] = name
    return assignment_map


def assign_pytorch_weights(pretrained_model_file, pretrained_prefix=None):
    """Assign pytorch weights to tf model."""
    import torch
    checkpoint = torch.load(pretrained_model_file)
    return assign_weights(checkpoint, pretrained_prefix)


def load_weight(pt_state_dict, vars):
    """Load weigths."""
    vars_values = []
    for idx, var in enumerate(vars):
        var_name = var.name
        op_name, transpose = convert_name(var_name)
        pt_name = list(pt_state_dict.keys())[idx]
        logging.info('{} ==> {}'.format(var_name, pt_name))
        values = pt_state_dict[pt_name].cpu().numpy()
        if transpose:
            values = numpy.transpose(values)
        if list(var.shape) != list(values.shape):
            raise ValueError("{}={} shape not equals {}={} shape".format(var.name, var.shape, pt_name, values.shape))
        vars_values.append((var, values))
    return vars_values


def assign_weights(pt_state_dict, pretrained_prefix=None):
    """Load pytorch state_dict and assign to tensorflow model."""
    import tensorflow as tf
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars.pop(0)
    pt_state_dict = {k: v for k, v in pt_state_dict.items() if 'num_batches_tracked' not in k}

    def _filter_vars_by_keys(var):

        for key in pretrained_prefix.keys():
            if var.name.startswith(key):
                return True

    def _filter_state_by_keys(state):
        for key in pretrained_prefix.values():
            if state.startswith(key):
                return True

    if pretrained_prefix:
        vars = list(filter(_filter_vars_by_keys, vars))
        pt_state_dict = {k: v for k, v in pt_state_dict.items() if _filter_state_by_keys(k)}
    if not pt_state_dict or not vars:
        raise ValueError("pertrained weight is None, please check the pretrained_prefix: {}".format(pretrained_prefix))
    vars_values = load_weight(pt_state_dict, vars)
    return [_var.assign(_value) for _var, _value in vars_values]


def convert_name(tf_name, start_prefix_to_remove=""):
    """Convert a TF variable name in a pytorch model weight name."""
    tf_name = tf_name.replace(":0", "")
    tf_name = re.sub(r"/[^/]*___([^/]*)/", r"/\1/", tf_name)
    tf_name = tf_name.replace("_._", "/")
    tf_name = re.sub(r"//+", "/", tf_name)
    tf_name = tf_name.split("/")
    tf_name = tf_name[1:]
    transpose = bool(tf_name[-1] == "kernel" or "emb_projs" in tf_name or "out_projs" in tf_name)
    if tf_name[-1] == "kernel" or tf_name[-1] == "embeddings" or tf_name[-1] == "gamma":
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"
    if tf_name[-1] == "moving_mean":
        tf_name[-1] = "running_mean"
    if tf_name[-1] == "moving_variance":
        tf_name[-1] = "running_var"
    # Remove prefix if needed
    tf_name = ".".join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, "", 1)
    return tf_name, transpose
