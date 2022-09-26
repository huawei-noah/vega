# -*- coding:utf-8 -*-

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

"""The trainer program for Auto Lane."""

import logging
import os
import time
import numpy as np
from pycocotools.coco import COCO
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.common.dtype as mstype
from mindspore.train import Model as MsModel
from mindspore import Tensor
from mindspore.nn import SGD
from vega.common import ClassFactory, ClassType
from vega.trainer.trainer_ms import TrainerMs
from vega.datasets.conf.dataset import DatasetConfig
from .src.model_utils.config import config
from .src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from .src.lr_schedule import dynamic_lr
from .src.network_define import WithLossCell, TrainOneStepCell, LossNet
from .src.util import coco_eval, bbox2result_1image, results2json

logger = logging.getLogger(__name__)


def valid():
    """Construct the trainer of SpNas."""
    config_val = DatasetConfig().to_dict()
    dataset_type = config_val.type
    config_val = config_val['_class_data'].val
    prefix = "FasterRcnn_eval.mindrecord"
    mindrecord_dir = config_val.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)

    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if dataset_type == "CocoDataset":
            if os.path.isdir(config_val.coco_root):
                data_to_mindrecord_byte_image(config_val, "coco", False, prefix, file_num=1)
            else:
                logging.info("coco_root not exits.")
        else:
            if os.path.isdir(config_val.IMAGE_DIR) and os.path.exists(config_val.ANNO_PATH):
                data_to_mindrecord_byte_image(config_val, "other", False, prefix, file_num=1)
            else:
                logging.info("IMAGE_DIR or ANNO_PATH not exits.")
    dataset = create_fasterrcnn_dataset(config_val, mindrecord_file, batch_size=config_val.test_batch_size,
                                        is_training=False)
    return dataset


def train():
    """Train fasterrcnn dataset."""
    config_train = DatasetConfig().to_dict()
    dataset_type = config_train.type
    config_train = config_train['_class_data'].train
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config_train.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")
    rank = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if dataset_type == "CocoDataset":
            if os.path.isdir(config_train.coco_root):
                if not os.path.exists(config_train.coco_root):
                    logging.info("Please make sure config:coco_root is valid.")
                    raise ValueError(config_train.coco_root)
                data_to_mindrecord_byte_image(config_train, "coco", True, prefix)
            else:
                logging.info("coco_root not exits.")
        else:
            if os.path.isdir(config_train.image_dir) and os.path.exists(config_train.anno_path):
                if not os.path.exists(config_train.image_dir):
                    logging.info("Please make sure config:image_dir is valid.")
                    raise ValueError(config_train.image_dir)
                data_to_mindrecord_byte_image(config_train, "other", True, prefix)
            else:
                logging.info("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)
    dataset = create_fasterrcnn_dataset(config_train, mindrecord_file, batch_size=config_train.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config_train.num_parallel_workers,
                                        python_multiprocessing=config_train.python_multiprocessing)
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
        self.valid_metrics = self._init_metrics()

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
        net = TrainOneStepCell(net_with_loss, self.optimizer, sens=config.loss_scale)

        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps, keep_checkpoint_max=1)
        save_path = self.get_local_worker_path(self.step_name, self.worker_id)
        ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
        loss_cb = LossMonitor(per_print_times=1)
        callback_list = [ckpoint_cb, loss_cb]
        self.ms_model = MsModel(net)
        try:
            self.ms_model.train(epoch=self.config.epochs,
                                train_dataset=dataset,
                                callbacks=callback_list,
                                dataset_sink_mode=False)
        except RuntimeError as e:
            logging.warning(f"failed to train the model, skip it, message: {str(e)}")

    def _valid_epoch(self):
        """Construct the trainer of SpNas."""
        dataset = self.valid_loader
        self.model.set_train(False)
        self.model.to_float(mstype.float16)
        outputs = []
        dataset_coco = COCO(self.config.metric.params.anno_path)

        max_num = 128
        for data in dataset.create_dict_iterator(num_epochs=1):

            img_data = data['image']
            img_metas = data['image_shape']
            gt_bboxes = data['box']
            gt_labels = data['label']
            gt_num = data['valid_num']
            output = self.model(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
            all_output_bbox = output[0]
            all_output_label = output[1]
            all_output_mask = output[2]

            for j in range(config.test_batch_size):
                all_output_bbox_squee = np.squeeze(all_output_bbox.asnumpy()[j, :, :])
                all_output_label_squee = np.squeeze(all_output_label.asnumpy()[j, :, :])
                all_output_mask_squee = np.squeeze(all_output_mask.asnumpy()[j, :, :])

                all_bboxes_tmp_mask = all_output_bbox_squee[all_output_mask_squee, :]
                all_labels_tmp_mask = all_output_label_squee[all_output_mask_squee]

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
