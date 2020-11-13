# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Calculate model forward latency."""
import time
import zeus
import numpy as np
from zeus.common.user_config import UserConfig


def calc_forward_latency(model, input, sess_config=None, num=100):
    """Model forward latency calculation.

    :param model: network model
    :type model: torch or tf module
    :param input: input tensor
    :type input: Tensor of torch or tf
    :param num: forward number
    :type num: int
    :return: forward latency
    :rtype: float
    """
    step_cfg = UserConfig().data.get("nas")
    if hasattr(step_cfg, "evaluator"):
        evaluate_cfg = step_cfg.get("evaluator")
        if hasattr(evaluate_cfg, "davinci_mobile_evaluator"):
            evaluate_config = evaluate_cfg.get("davinci_mobile_evaluator")
            latency = _calc_forward_latency_davinci(model, input, sess_config, evaluate_config)
    else:
        latency = _calc_forward_latency_gpu(model, input, sess_config, num)
    return latency


def _calc_forward_latency_gpu(model, input, sess_config=None, num=100):
    """Model forward latency calculation.

    :param model: network model
    :type model: torch or tf module
    :param input: input tensor
    :type input: Tensor of torch or tf
    :param num: forward number
    :type num: int
    :return: forward latency
    :rtype: float
    """
    prepare_num = int(0.05 * num)
    if zeus.is_torch_backend():
        for _ in range(prepare_num):
            model(input)
        start_time = time.time()
        for _ in range(num):
            model(input)
        latency = (time.time() - start_time) / num
    elif zeus.is_tf_backend():
        import tensorflow.compat.v1 as tf
        with tf.Graph().as_default() as graph:
            input_holder = tf.placeholder(dtype=tf.float32, shape=input.shape.as_list())
            model.training = False
            output = model(input_holder)
            with tf.Session(config=sess_config) as sess:
                sess.run(tf.global_variables_initializer())
                input = tf.random.uniform(input.shape.as_list(), dtype=input.dtype)
                input_numpy = input.eval(session=sess)
                for _ in range(prepare_num):
                    sess.run(output, feed_dict={input_holder: input_numpy})
                start_time = time.time()
                for _ in range(num):
                    sess.run(output, feed_dict={input_holder: input_numpy})
        latency = (time.time() - start_time) / num
    elif zeus.is_ms_backend():
        latency = 0.
    return latency


def _calc_forward_latency_davinci(model, input, sess_config=None, num=1, evaluate_config=None):
    """Model forward latency calculation.

    :param model: network model
    :type model: torch or tf module
    :param input: input tensor
    :type input: Tensor of torch or tf
    :param num: forward number
    :type num: int
    :param evaluate_config: some config for evaluate in davinci
    :type evaluate_config: dict
    :return: forward latency
    :rtype: float
    """
    from zeus.evaluator.tools.evaluate_davinci_bolt import evaluate
    # backend = evaluate_config.get("backend")
    hardware = evaluate_config.get("hardware")
    remote_host = evaluate_config.get("remote_host")

    save_data_file = "./input.bin"
    latency = 0.
    if zeus.is_torch_backend():
        import torch
        input_shape = input.shape
        if torch.is_tensor(input):
            input = input.numpy()
        input.tofile(save_data_file)
        for _ in range(num):
            results = evaluate("pytorch", hardware, remote_host, model, None, save_data_file, input_shape)
            latency += np.float(results.get("latency"))
    return latency / num
