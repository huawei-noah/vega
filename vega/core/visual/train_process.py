# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Save Train processing info."""
import logging
from tensorboardX import SummaryWriter
from vega.core.common import FileOps


def dump_trainer_visual_info(trainer, epoch, visual_data):
    """Dump triner info to tensorboard event files.

    :param trainer: trainer.
    :type worker: object that the class was inherited from DistributedWorker.
    :param epoch: number of epoch.
    :type epoch: int.
    :param visual_data: train's visual data.
    :type visual_data: ordered dictionary.

    """
    (visual, _, _, title, worker_id, output_path) = _get_trainer_info(trainer)
    if visual is not True:
        return
    prefix_name = "{}".format(worker_id)
    lines = {}
    for _name, data in visual_data.items():
        line_name = "{}.{}".format(prefix_name, _name)
        lines[line_name] = data
    if len(lines) > 0:
        writer = SummaryWriter(output_path, comment=title)
        writer.add_scalars(title, lines, epoch)
        writer.close()


def dump_model_visual_info(trainer, epoch, model, inputs):
    """Dump model to tensorboard event files.

    :param trainer: trainer.
    :type worker: object that the class was inherited from DistributedWorker.
    :param model: model.
    :type model: model.
    :param inputs: input data.
    :type inputs: data.

    """
    (_, visual, interval, title, worker_id, output_path) = _get_trainer_info(trainer)
    if visual is not True:
        return
    if epoch % interval != 0:
        return
    title = str(worker_id)
    _path = FileOps.join_path(output_path, title)
    FileOps.make_dir(_path)
    try:
        with SummaryWriter(_path) as writer:
            writer.add_graph(model, (inputs,))
    except Exception as e:
        logging.error("Failed to dump model visual info, worker id: {}, epoch: {}, error: {}".format(
            worker_id, epoch, str(e)
        ))


def _get_trainer_info(trainer):
    if "visualize" in trainer.cfg.keys():
        interval = trainer.cfg.visualize.model.interval
        visual_process = trainer.cfg.visualize.train_process.visual
        visual_model = trainer.cfg.visualize.model.visual
    else:
        interval = 10
        visual_process = True
        visual_model = True
    worker_id = trainer.worker_id
    title = trainer.cfg.step_name
    output_path = trainer.local_visual_path
    return visual_process, visual_model, interval, title, worker_id, output_path
