# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for Auto Lane."""

import logging
import os
import time
import numpy as np
from pycocotools.coco import COCO
from vega.common import ClassFactory, ClassType
from vega.trainer.trainer_ms import TrainerMs
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.common.dtype as mstype
from mindspore.train import Model as MsModel
from mindspore import Tensor
from mindspore.nn import SGD
from .src.model_utils.config import config
from .src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from .src.lr_schedule import dynamic_lr
from .src.network_define import WithLossCell, TrainOneStepCell, LossNet
from .src.util import coco_eval, bbox2result_1image, results2json
from vega.datasets.conf.dataset import DatasetConfig

logger = logging.getLogger(__name__)


def valid():
    """Construct the trainer of SpNas."""
    config = DatasetConfig().to_dict()
    config = config['_class_data'].val
    prefix = "FasterRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)

    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                data_to_mindrecord_byte_image(config, "coco", False, prefix, file_num=1)
            else:
                logging.info("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                data_to_mindrecord_byte_image(config, "other", False, prefix, file_num=1)
            else:
                logging.info("IMAGE_DIR or ANNO_PATH not exits.")
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.test_batch_size, is_training=False)
    return dataset


def train():
    """Train fasterrcnn dataset."""
    config = DatasetConfig().to_dict()
    config = config['_class_data'].train
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")
    rank = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                if not os.path.exists(config.coco_root):
                    logging.info("Please make sure config:coco_root is valid.")
                    raise ValueError(config.coco_root)
                data_to_mindrecord_byte_image(config, "coco", True, prefix)
            else:
                logging.info("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                if not os.path.exists(config.image_dir):
                    logging.info("Please make sure config:image_dir is valid.")
                    raise ValueError(config.image_dir)
                data_to_mindrecord_byte_image(config, "other", True, prefix)
            else:
                logging.info("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)
    return dataset


@ClassFactory.register(ClassType.TRAINER)
class SpNasTrainerCallback(TrainerMs):
    """Construct the trainer of SpNas."""

    disable_callbacks = ['ProgressLogger']

    def build(self):
        """Construct the trainer of SpNas."""
        logging.debug("Trainer Config: {}".format(self.config))
        self._init_hps()
        self.use_syncbn = self.config.syncbn
        if not self.train_loader:
            self.train_loader = train()
        if not self.valid_loader:
            self.valid_loader = valid()
        self.batch_num_train = self.train_loader.get_dataset_size()
        self.batch_num_valid = self.valid_loader.get_dataset_size()

    def _train_epoch(self):
        """Construct the trainer of SpNas."""
        dataset = self.train_loader
        dataset_size = dataset.get_dataset_size()
        self.model = self.model.set_train()
        self.model.to_float(mstype.float16)
        self.loss = LossNet()
        lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
        self.optimizer = SGD(params=self.model.trainable_params(), learning_rate=lr, momentum=config.momentum,
                             weight_decay=config.weight_decay, loss_scale=config.loss_scale)
        net_with_loss = WithLossCell(self.model, self.loss)
        self.model = TrainOneStepCell(net_with_loss, self.optimizer, sens=config.loss_scale)

        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps, keep_checkpoint_max=1)
        save_path = self.get_local_worker_path(self.step_name, self.worker_id)
        ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
        loss_cb = LossMonitor(per_print_times=1)
        callback_list = [ckpoint_cb, loss_cb]
        self.ms_model = MsModel(self.model)
        try:
            self.ms_model.train(epoch=self.trainer.epochs,
                                train_dataset=dataset,
                                callbacks=callback_list,
                                dataset_sink_mode=False)
        except RuntimeError as e:
            logging.warning(f"failed to train the model, skip it, message: {str(e)}")

    def _valid_epoch(self):
        """Construct the trainer of SpNas."""
        dataset = self.valid_loader
        self.model.set_train(False)
        outputs = []
        dataset_coco = COCO(config.ann_file)

        max_num = 128
        for data in dataset.create_dict_iterator(num_epochs=1):

            img_data = data['image']
            img_metas = data['image_shape']
            gt_bboxes = data['box']
            gt_labels = data['label']
            gt_num = data['valid_num']
            output = self.model(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
            all_bbox = output[0]
            all_label = output[1]
            all_mask = output[2]

            for j in range(config.test_batch_size):
                all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
                all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
                all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

                all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
                all_labels_tmp_mask = all_label_squee[all_mask_squee]

                if all_bboxes_tmp_mask.shape[0] > max_num:
                    inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                    inds = inds[:max_num]
                    all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                    all_labels_tmp_mask = all_labels_tmp_mask[inds]

                outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)

                outputs.append(outputs_tmp)

        eval_types = ["bbox"]
        result_files = results2json(dataset_coco, outputs, "./results.pkl")
        metrics = coco_eval(result_files, eval_types, dataset_coco, single_result=True)
        self.valid_metrics.update(metrics)
        valid_logs = dict()
        valid_logs['cur_valid_perfs'] = self.valid_metrics.results
        self.callbacks.after_valid(valid_logs)
