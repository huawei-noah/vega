# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE
"""
Get vision by tensorboard.

    usage:
        1. tensorboard --logdir=/tmp/.xt_data/tensorboard
        2. and then, open chrome with url: http://YOUR.SERVER.IP:6006
        3. if multi-scaler, you NEED re-run step-1 above!!!  bugs
"""
import os
from datetime import datetime
from time import sleep
import numpy as np
import shutil
from absl import logging
from tensorboardX import SummaryWriter


def is_board_running(pro_name="tensorboard"):
    """Check if process running."""
    cmd = ('ps aux | grep "' + pro_name + '" | grep -v grep | grep -v tail | grep -v keepH5ssAlive')
    try:
        process_num = len(os.popen(cmd).readlines())
        if process_num >= 1:
            return True
        else:
            return False
    except BaseException as err:
        logging.warning("check process failed with {}.".format(err))
        return False


def clean_board_dir(_to_deleta_dir):
    """Re-clear tensorboard dir."""
    if os.path.isdir(_to_deleta_dir):
        shutil.rmtree(_to_deleta_dir, ignore_errors=True)
        print("will clean path: {} for board...".format(_to_deleta_dir))
        sleep(0.01)


class SummaryBoard(object):
    """SummaryBoard used for the visual base the tensorboardX."""

    def __init__(self, archive_root, fixed_path=None):
        """Init the summaryBoard, fixed_path could refer to worker_id."""
        self._archive = archive_root
        if not os.path.isdir(archive_root):
            os.makedirs(archive_root)
        if not fixed_path:
            self.logdir = os.path.join(
                archive_root, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            self.logdir = os.path.join(archive_root, str(fixed_path))
        self.writer = SummaryWriter(logdir=self.logdir)

    def insert_records(self, records):
        """Insert records."""
        for record in records:
            name, value, index = record
            if np.isnan(value) or not value:
                continue
            self.writer.add_scalar(name, value, index)
        self.writer.flush()

    def insert_epoch_logs(self, logs, epoch):
        """Insert logs after epoch."""
        for k, v in logs:
            if not v:
                continue
            self.add_scalar(k, v, epoch, flush=False)
        self.writer.flush()

    def add_scalar(self, name, value, index, walltime=None, flush=False):
        """Add scalar func."""
        if walltime is not None:
            self.writer.add_scalar(name, value, index, walltime=walltime)
        else:
            self.writer.add_scalar(name, value, index)

        if flush:
            self.writer.flush()

    def add_graph(self, model=None, graph=None, feed_data=None, backend=None):
        """Add graph."""
        if backend == "tf":
            self._add_tf_graph(graph)

        elif backend == "torch":
            self._add_torch_graph(model, feed_data)

        elif backend == "ms":
            self._add_ms_graph()
        else:
            print("Add graph failed with non-known backend!")

    def _add_tf_graph(self, graph):
        import tensorflow as tf
        with graph.as_default():
            writer = tf.summary.FileWriter(
                logdir=os.path.join(self.logdir, "model_def"), graph=graph)
            writer.flush()
            writer.close()

    def _add_torch_graph(self, model, feed_data):
        self.writer.add_graph(model, (feed_data,))

    def _add_ms_graph(self):
        pass

    def close(self):
        """Close SummaryBoard, contains file.close and process shutdown."""
        self.writer.flush()
        self.writer.close()
