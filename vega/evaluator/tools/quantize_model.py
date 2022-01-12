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

"""AMCT quantize functions."""
import os
import shutil
import logging
import numpy as np
import tensorflow as tf
import amct_tensorflow as amct


def quantize_model(output_graph_file, test_data, input_holder, output):
    """Quantize tf model by using amct tool."""
    batch = load_image(test_data, input_holder.shape)
    input_name = input_holder.name
    output_name = output.name
    with tf.io.gfile.GFile(output_graph_file, mode='rb') as model:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(model.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    input_tensor = graph.get_tensor_by_name(input_name)
    output_tensor = graph.get_tensor_by_name(output_name)

    base_dir = os.path.dirname(output_graph_file)
    config_path = os.path.join(base_dir, 'config.json')
    amct.create_quant_config(config_file=config_path,
                             graph=graph,
                             skip_layers=[],
                             batch_num=1)
    record_path = os.path.join(base_dir, 'record.txt')
    amct.quantize_model(graph=graph,
                        config_file=config_path,
                        record_file=record_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(output_tensor, feed_dict={input_tensor: batch})
    save_path = os.path.join(base_dir, 'tf_model')
    amct.save_model(pb_model=output_graph_file,
                    outputs=[output_name[:-2]],
                    record_file=record_path,
                    save_path=save_path)
    shutil.copy('{}_quantized.pb'.format(save_path), output_graph_file)
    logging.info('amct quantinize successfully.')


def load_image(test_data, shape):
    """Load calibration images."""
    test_np = np.fromfile(test_data, dtype=np.float32)
    test_shape = (-1,) + tuple(shape[1:])
    test_np = np.reshape(test_np, test_shape)
    calib_num = 32
    if test_np.shape[0] > calib_num:
        return test_np[:calib_num]
    else:
        return test_np
