# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for SP-NAS."""

import logging
import os
import subprocess
import sys
import torch
import time
import datetime
import shutil

from vega.algorithms.nas.sp_nas.utils import dict_to_json, extract_backbone_from_pth, update_config, ListDict
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.pytorch import Trainer
from vega.core.trainer.utils import WorkerTypes
from vega.core.common import Config, FileOps
from vega.algorithms.nas.sp_nas.spnet import *


@ClassFactory.register(ClassType.TRAINER)
class SpNasTrainer(Trainer):
    """Construct the trainer of SP-NAS."""

    def __init__(self, sample, id):
        """Construct the SPNAS class.

        :param sample: the config of model and the sample info
        :type sample: tuple(config, dict)
        :param id: worker id
        :type id: int
        """
        super(SpNasTrainer, self).__init__(self.cfg)
        logging.info("init BackboneNasTrainer")
        self.worker_type = WorkerTypes.TRAINER
        self._worker_id = id
        self.gpus = self.cfg.gpus
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self._train_script = os.path.join(dir_path, 'tools/dist_train.sh')
        self._eval_script = os.path.join(dir_path, 'tools/test.py')
        self.sample_result = sample

    @property
    def cfg_path(self):
        """Get local config path."""
        return os.path.join(self.get_local_worker_path(), 'config.json')

    def _sample_to_config(self):
        """Save model config to local config file."""
        if self.sample_result is not None:
            config = self.sample_result[0]
            logging.info("Sampling architecture: {}".format(self.sample_result[1]))
            pre_worker_id = self.sample_result[1]['pre_worker_id']
            pre_arch = self.sample_result[1]['pre_arch']
            if pre_worker_id >= 0:
                # self.get_worker_result_path('group', pre_worker_id)
                config['model']['pretrained'] = extract_backbone_from_pth(self.local_output_path,
                                                                          pre_worker_id, pre_arch)
            # record sample results
            self.sample_results = dict(arch=self.sample_result[1]['arch'],
                                       worker_id=self._worker_id,
                                       pre_arch=pre_arch,
                                       pre_worker_id=pre_worker_id)
        else:
            config = self._init_model()
        # write config dict to cfg_path.
        dict_to_json(config, self.cfg_path)

    def _init_model(self):
        """Initialize model if fully training a model.

        :return: config of fully train model
        :rtype: config file
        """
        config = Config(self.cfg.config_template)
        config['total_epochs'] = self.cfg.epoch
        if 'model_desc_file' in self.cfg:
            self.download_task_folder()
            _total_list = ListDict.load_csv(os.path.join(self.local_output_path,
                                                         self.cfg.model_desc_file))
            pre_arch = _total_list.sort('mAP')[0]['arch']
            pretrained = pre_arch.split('_')[1]
            pre_worker_id = _total_list.sort('mAP')[0]['pre_worker_id']
            model_desc = dict(arch=pre_arch,
                              pre_arch=pretrained,
                              pre_worker_id=-1)
            logging.info("Initialize fully train model from: {}".format(model_desc))
            if self.cfg.regnition:
                # re-write config from previous result
                config['model']['backbone']['reignition'] = True
                config['model']['pretrained'] = os.path.join(self.local_output_path,
                                                             pretrained + '_imagenet.pth')
            else:
                config['model']['pretrained'] = extract_backbone_from_pth(self.local_output_path,
                                                                          pre_worker_id, pretrained)

        elif 'model_desc' in self.cfg:
            model_desc = self.cfg.model_desc
        else:
            raise ValueError('Missing model description!')
        model_desc = update_config(config, model_desc)
        return model_desc

    def train_process(self):
        """Construct the whole train process of the TrainWorker specified in configuration."""
        logging.info("start training")
        self._sample_to_config()

        t_start = time.time()
        memory = self._train()
        self._valid()
        t_end = time.time()

        results = self.collect_results(memory, t_start, t_end)
        self._save_performance(results)
        self._backup()

    def _train(self):
        """Construct the train process of the SPNet.

        :return: the memory of model on training set
        :rtype: int
        """
        logging.info('*' * 30 + "worker-{}: Start Training".format(self._worker_id) + '*' * 30)
        torch.cuda.empty_cache()
        mem = 0
        try:
            cmd = ['bash', self._train_script, self.cfg_path, str(self.gpus), '--work_dir',
                   self.get_local_worker_path()]
            sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in sp.stdout:
                sys.stdout.write(line.decode('utf-8'))
                if 'RuntimeError: CUDA out of memory.' in line.decode():
                    raise subprocess.CalledProcessError(1, cmd, output='RuntimeError: CUDA out of memory.')
                if 'memory: ' in line.decode():
                    cur_m = int(line.decode().split('memory: ')[1].split(',')[0])
                    if cur_m > mem:
                        mem = cur_m
            retcode = sp.wait()
            if retcode:
                raise subprocess.CalledProcessError(retcode, cmd)
        except subprocess.CalledProcessError as error:
            if error.output:
                logging.info("RuntimeError: CUDA out of memory.")
            else:
                raise error
        return mem

    def _valid(self):
        """Construct the valiation process of the SPNet."""
        logging.info('*' * 30 + "worker-{}: Start Evaluation".format(self._worker_id) + '*' * 30)
        try:
            cmd = ['python3', self._eval_script, self.cfg_path, '--work_dir', self.get_local_worker_path()]
            sp = subprocess.Popen(cmd)
            retcode = sp.wait()
            if retcode:
                raise subprocess.CalledProcessError(retcode, cmd)
        except subprocess.CalledProcessError as error:
            raise error

    def collect_results(self, memory, t_start, t_end):
        """Collect performance results.

        :param memory: the need memory during training
        :type memory: int
        :param t_start: the timestamp of start the training
        :type t_start: float
        :param t_end: the timestamp of end the evaluation
        :type t_end: float
        :return: performance results
        :rtype: dict
        """
        import pickle as pkl
        mAP_file = os.path.join(self.get_local_worker_path(), 'mAP.pkl')
        fps_file = os.path.join(self.get_local_worker_path(), 'fps.pkl')
        # get mAP
        if os.path.isfile(mAP_file):
            with open(mAP_file, 'rb') as f:
                mAP = pkl.load(f)['bbox'][0]
        else:
            mAP = -1
            logging.info('mAP record file {} not found. Evaluation fails.'.format(mAP_file))
        # get fps
        if os.path.isfile(fps_file):
            with open(fps_file, 'rb') as f:
                fps = pkl.load(f)['model']
        else:
            fps = -1
            logging.info('fps record file {} not found. Evaluation fails.'.format(fps_file))

        run_time = dict(t_start=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(t_start)),
                        t_end=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(t_end)),
                        t_eclapsed=' ' + str(datetime.timedelta(seconds=(t_end - t_start))))
        results = self.sample_results
        results.update(run_time)
        results.update(dict(memory=memory, fps=fps, mAP=mAP))
        return results

    def _save_performance(self, results):
        """Save performance into performance.pkl and save checkpoint to output_dir.

        :param results: performance results
        :type sr: dict
        """
        logging.info("performance=%s", str(results))
        performance_dir = os.path.join(self.get_local_worker_path(), 'performance')
        FileOps.make_dir(performance_dir)
        FileOps.dump_pickle(results, os.path.join(performance_dir, 'performance.pkl'))
        logging.info("performance save to %s", performance_dir)
        # copy pth to output dir
        output_dir = os.path.join(self.local_output_path, str(self._worker_id))
        FileOps.make_dir(output_dir)
        shutil.copy(os.path.join(self.get_local_worker_path(), 'latest.pth'),
                    os.path.join(output_dir, results['arch'].split('_')[1] + '.pth'))
        logging.info("Latest checkpoint save to %s", output_dir)
