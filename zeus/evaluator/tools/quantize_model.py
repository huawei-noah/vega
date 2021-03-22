# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""AMCT quantize functions."""
import os
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
    os.system('cp {}_quantized.pb {}'.format(save_path, output_graph_file))
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
