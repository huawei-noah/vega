# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Visual callback definition."""
import os
import logging
import zeus
import numpy as np
from copy import deepcopy
from .callback import Callback
from zeus.common import ClassFactory, ClassType
from zeus.common import TaskOps
from zeus.visual.tensorboarder import SummaryBoard


def _flat_items(data, parents=tuple()):
    for k, v in data.items():
        try:
            yield from _flat_items(v, parents=parents + (k,))
        except AttributeError:
            yield parents + (k,), v


def make_keys_readable(records):
    """Make keys readable with flat&join."""
    return [("/".join(k), v) for k, v in _flat_items(records)]


@ClassFactory.register(ClassType.CALLBACK)
class VisualCallBack(Callback):
    """Callback that write the records for visual."""

    def __init__(self):
        """Initialize Visual callback."""
        super(VisualCallBack, self).__init__()
        self.priority = 290
        self._archive_root = TaskOps().local_visual_path
        self._fix_path = None
        self.summary = None
        self.writer = None

        self.input = None
        self.model = None

        self._need_keys = {"loss_avg", "lr"}
        self._info = {k: 0. for k in self._need_keys}

    def before_train(self, logs=None):
        """Fetch trainer info before train stage."""
        self._fix_path = "_".join([self.trainer.step_name, str(self.trainer.worker_id)])
        self.summary = SummaryBoard(self._archive_root, self._fix_path)

        # add graph only once.
        if zeus.is_tf_backend():
            import tensorflow as tf
            datasets = self.trainer.valid_input_fn()
            data_iter = tf.compat.v1.data.make_one_shot_iterator(datasets)
            input_data, _ = data_iter.get_next()
            self.input = input_data[:1]

            graph = self.trainer.graph
            _graph_name_list = [n.name for n in graph.as_graph_def().node]
            if len(_graph_name_list) < 2:
                graph = _fetch_tf_graph(self.trainer.model, self.input)

            self.summary.add_graph(graph=graph, backend="tf")
        elif zeus.is_torch_backend():
            model = self.trainer.model
            data_iter = iter(self.trainer.train_loader)
            input_batch, _ = data_iter.next()

            input_data = input_batch[:1]
            if self.trainer.use_cuda and not self.trainer.config.is_detection_trainer:
                input_data = input_data.cuda()
            try:
                self.summary.add_graph(model=model, feed_data=input_data,
                                       backend="torch")
            except BaseException as err:
                logging.warning("Dump PyTorch model failed! with: \n{}".format(err))

        elif zeus.is_ms_backend():
            logging.debug("Don't support mindspore model dump yet.")
        else:
            logging.warning("non-known backend.")

    def after_epoch(self, epoch, logs=None):
        """Collect data after epoch, and 'after_epoch' data could contains 'after_valid'."""
        readable_records = make_keys_readable(logs)
        self.summary.insert_epoch_logs(readable_records, epoch)

        # update info
        info_records = [("/".join(["info", k]), self._info[k]) for k in self._need_keys]
        self.summary.insert_epoch_logs(info_records, epoch)

    def after_valid(self, logs=None):
        """Check records after valid."""
        pass

    def after_train_step(self, batch_index, logs=None):
        """Collect info after each train step."""
        if not logs:
            return
        for _k in self._need_keys:
            self._info.update({_k: logs.get(_k, 0.)})

    def _need_record_graph(self):
        """Record graph within 'train' stage."""
        return "train" in self._fix_path

    def after_train(self, logs=None):
        """Shutdown summary after train."""
        self.summary.close()


def _fetch_tf_graph(model, input):
    import tensorflow as tf
    graph = tf.Graph()
    with graph.as_default():

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        dummy_input = tf.placeholder(dtype=tf.float32, shape=input.shape.as_list())
        model.training = True
        out = model(dummy_input)
        sess.run(tf.global_variables_initializer())

        o = sess.run(out, feed_dict={dummy_input: np.ones(input.shape.as_list())})
        # print(np.shape(o), o)
    return graph
