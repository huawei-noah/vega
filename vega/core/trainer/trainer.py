# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer."""
import logging
import os
import pickle
import vega
import glob
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.config import Config, obj2config
from vega.core.trainer.distributed_worker import DistributedWorker
from vega.core.trainer.utils import WorkerTypes
from vega.core.trainer.conf import TrainerConfig
from vega.search_space.networks import NetworkDesc
from vega.core.common import FileOps, init_log
from vega.core.trainer.callbacks import CallbackList
from vega.core.trainer.modules.optimizer import Optimizer
from vega.core.trainer.modules.lr_schedulers import LrScheduler
from vega.core.trainer.modules.losses import Loss
from vega.core.common.loader import load_conf_from_desc

if vega.is_torch_backend():
    import torch
    from vega.core.metrics.pytorch.metrics import Metrics

    try:
        import horovod.torch as hvd
    except Exception:
        # logging.warning("horovod not been installed, {}".format(str(e)))
        pass
    try:
        import apex
        from apex import amp
    except Exception:
        # logging.warning("apex not been installed, {}".format(str(e)))
        pass
elif vega.is_tf_backend():
    import tensorflow as tf
    from tensorflow.python.estimator import estimator as est
    from vega.core.metrics.tensorflow.metrics import Metrics

    try:
        import horovod.tensorflow as hvd
    except Exception:
        # logging.warning("horovod not been installed, {}".format(str(e)))
        pass
if vega.is_npu_device():
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator import npu_ops
    from hccl.manage.api import get_local_rank_id
    from hccl.manage.api import get_rank_size
    from hccl.manage.api import get_rank_id
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.TRAINER)
class Trainer(DistributedWorker):
    """Trainer class.

    :param model: input model, defaults to None
    :type model: tf model, optional
    :param id: id of the model, defaults to None
    :type id: int, optional
    :param hps: hyperparameters, defaults to None
    :type hps: dict, optional
    """

    # __worker_id__ = 0
    config = TrainerConfig()

    def __init__(self, model=None, id=None, hps=None, load_ckpt_flag=False, **kwargs):
        super(Trainer, self).__init__()
        self.worker_type = WorkerTypes.TRAINER
        Trainer.__worker_id__ += 1
        if id is not None:
            self._worker_id = id
        else:
            self._worker_id = Trainer.__worker_id__

        # Data Memeber list of Trainer
        self.is_chief = True
        self.use_cuda = self.config.cuda
        self.epochs = self.config.epochs
        self.do_validation = True
        self.auto_save_ckpt = True
        self.auto_save_perf = True
        self.skip_train = False
        self.valid_interval = self.config.valid_interval
        self.hps = hps
        self.model = model
        self.optimizer = None
        self.lr_scheduler = None
        self.loss = None
        self.use_syncbn = self.config.syncbn
        self.use_amp = self.config.amp
        self.train_metrics = None
        self.valid_metrics = None
        self.call_metrics_on_train = self.config.call_metrics_on_train
        self.train_loader = None
        self.valid_loader = None
        self.train_step = None
        self.valid_step = None
        self.make_batch = None
        self.model_fn = None
        self.train_input_fn = None
        self.valid_input_fn = None
        self.callbacks = None
        self.performance = None
        self.model_desc = {}
        self.visual_data = {}
        self.load_ckpt_flag = load_ckpt_flag
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        self.model_path = FileOps.join_path(
            self.get_local_worker_path(), self.model_pickle_file_name)
        self.checkpoint_file = FileOps.join_path(
            self.get_local_worker_path(), self.checkpoint_file_name)
        self.weights_file = FileOps.join_path(
            self.get_local_worker_path(), "model_{}.pth".format(self.worker_id))
        self.distributed = self.config.distributed
        # Used by TimmTrainerCallbacks since it builds its trainer in
        # the before_train callback
        self.lazy_built = self.config.lazy_built
        # Indicate whether the necessary components of a trainer
        # has been built for running
        self.has_built = False
        self._world_size = 1
        self._rank_id = 0
        self._local_rank_id = 0
        self.config.kwargs = kwargs

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        init_log(log_file="worker_{}.txt".format(self.worker_id))
        logging.debug("Use the unified Trainer")

        if not self.lazy_built:
            self.build(model=self.model, hps=self.hps, load_ckpt_flag=self.load_ckpt_flag)
        self._init_callbacks(self.callbacks)
        self._train_loop()

    def build(self, model=None, optimizer=None, loss=None,
              lr_scheduler=None, metrics=None, hps=None,
              callbacks=None, train_loader=None, valid_loader=None,
              make_batch=None, train_step=None, valid_step=None,
              model_fn=None, train_input_fn=None, valid_input_fn=None,
              load_ckpt_flag=False,
              checkpoint_file_name="checkpoint.pth",
              model_pickle_file_name="model.pkl"):
        """Build the trainer by assembling the necessary components."""
        # Intitialize hyperparameters by parameters or configurations
        self._init_hps(hps)
        logging.debug("Trainer Config: {}".format(obj2config(self.config)))
        self.checkpoint_file_name = checkpoint_file_name
        self.model_pickle_file_name = model_pickle_file_name
        if vega.is_torch_backend():
            self._init_step_functions(make_batch, train_step, valid_step)
        elif vega.is_tf_backend():
            self._init_estimator_fn(model_fn, train_input_fn, valid_input_fn)
        self._init_tf_session()
        self._init_distributed_setting()
        self._init_cuda_setting()
        self._init_tf_estimator()
        self.do_validation = self.config.with_valid
        self.model = self._init_model(model)
        self.load_ckpt_flag = load_ckpt_flag
        if self.load_ckpt_flag:
            self.load_checkpoint()
        else:
            self._load_pretrained_model()
        self.use_syncbn = self.config.syncbn
        if self.use_syncbn and vega.is_torch_backend():
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.train_loader = self._init_dataloader(
            mode='train', loader=train_loader)
        self.valid_loader = self._init_dataloader(
            mode='val', loader=valid_loader)
        if vega.is_torch_backend():
            self.optimizer = Optimizer()(model=self.model, distributed=self.distributed) \
                if optimizer is None else optimizer
            self.loss = Loss()() if loss is None else loss
            self.lr_scheduler = LrScheduler()(self.optimizer) if lr_scheduler is None else lr_scheduler
        # Some trainer has different train batch size from valid batch
        self.train_metrics = self._init_metrics(metrics) if vega.is_torch_backend() else None
        self.valid_metrics = self._init_metrics(metrics)

        self._init_horovod_setting()
        if self.use_amp and vega.is_torch_backend():
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')
        if self.callbacks is None:
            self.callbacks = callbacks
        # self.output_model_desc()
        cur_working_dir = FileOps.join_path(self.local_output_path, self.step_name)
        FileOps.make_dir(cur_working_dir)
        # Make sure Trainer has been built for training
        self.has_built = True

    def _init_cuda_setting(self):
        """Init CUDA setting."""
        if not vega.is_torch_backend():
            return
        if not self.config.cuda:
            self.config.device = -1
            return
        self.config.device = self.config.cuda if self.config.cuda is not True else 0
        self.use_cuda = True
        if self.distributed:
            torch.cuda.set_device(self._local_rank_id)
        torch.cuda.manual_seed(self.config.seed)

    def _init_distributed_setting(self):
        if not self.distributed:
            return
        if vega.is_npu_device():
            self.npu_init = npu_ops.initialize_system()
            self.npu_shutdown = npu_ops.shutdown_system()
            self.sess.run(self.npu_init)
        self._world_size = hvd.size() if vega.is_gpu_device() else get_rank_size()
        self._rank_id = hvd.rank() if vega.is_gpu_device() else get_rank_id()
        self._local_rank_id = hvd.local_rank() if vega.is_gpu_device() else get_local_rank_id()

    def _init_horovod_setting(self):
        """Init horovod setting."""
        self.is_chief = True
        if self.distributed and vega.is_torch_backend():
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            if hvd.rank() != 0:
                self.is_chief = False
            else:
                self.is_chief = True

    def _init_hps(self, hps=None):
        """Load hps from file."""
        if hps is not None:
            self.hps = hps
        elif self.config.hps_file is not None:
            desc_file = self.config.hps_file.replace("{local_base_path}", self.local_base_path)
            self.hps = Config(desc_file)
        elif self.config.hps_folder is not None:
            folder = self.config.hps_folder.replace("{local_base_path}", self.local_base_path)
            pattern = FileOps.join_path(folder, "desc_*.json")
            desc_file = glob.glob(pattern)[0]
            self.hps = Config(desc_file)
        if self.hps and self.hps.get('trainer'):
            load_conf_from_desc(self.config, self.hps.get('trainer'))

    def _init_model(self, model=None):
        """Load model desc from save path and parse to model."""
        if model is not None:
            if vega.is_torch_backend() and self.use_cuda:
                model = model.cuda()
            return model
        model_cfg = Config(ClassFactory.__configs__.get('model'))
        if "model_desc_file" in model_cfg and model_cfg.model_desc_file is not None:
            desc_file = model_cfg.model_desc_file
            desc_file = desc_file.replace("{local_base_path}", self.local_base_path)
            if ":" not in desc_file:
                desc_file = os.path.abspath(desc_file)
            if ":" in desc_file:
                local_desc_file = FileOps.join_path(
                    self.local_output_path, os.path.basename(desc_file))
                FileOps.copy_file(desc_file, local_desc_file)
                desc_file = local_desc_file
            model_desc = Config(desc_file)
            logging.info("net_desc:{}".format(model_desc))
        elif "model_desc" in model_cfg and model_cfg.model_desc is not None:
            model_desc = model_cfg.model_desc
        elif "models_folder" in model_cfg and model_cfg.models_folder is not None:
            folder = model_cfg.models_folder.replace("{local_base_path}", self.local_base_path)
            pattern = FileOps.join_path(folder, "desc_*.json")
            desc_file = glob.glob(pattern)[0]
            model_desc = Config(desc_file)
        else:
            return None
        if model_desc is not None:
            self.model_desc = model_desc
            net_desc = NetworkDesc(model_desc)
            model = net_desc.to_model()
            if vega.is_torch_backend() and self.use_cuda:
                model = model.cuda()
            return model
        else:
            return None

    def _load_pretrained_model(self):
        if self.model is None:
            return
        if self.config.pretrained_model_file is not None:
            model_file = self.config.pretrained_model_file
            model_file = os.path.abspath(model_file)
            if vega.is_torch_backend():
                ckpt = torch.load(model_file)
                self.model.load_state_dict(ckpt)
            elif vega.is_tf_backend():
                model_folder = os.path.dirname(model_file)
                FileOps.copy_folder(model_folder, self.get_local_worker_path())
            return

    def load_checkpoint(self, worker_id=None, step_name=None, saved_folder=None):
        """Load checkpoint."""
        if saved_folder is None:
            if worker_id is None:
                worker_id = self.worker_id
            if step_name is None:
                step_name = self.step_name
            saved_folder = self.get_local_worker_path(step_name, worker_id)
        checkpoint_file = FileOps.join_path(
            saved_folder, self.checkpoint_file_name)
        model_pickle_file = FileOps.join_path(
            saved_folder, self.model_pickle_file_name)
        try:
            with open(model_pickle_file, 'rb') as f:
                model = pickle.load(f)
                if vega.is_torch_backend():
                    ckpt = torch.load(
                        checkpoint_file, map_location=torch.device('cpu'))
                    model.load_state_dict(ckpt['weight'])
                    if self.config.cuda:
                        model = model.cuda()
                elif vega.is_tf_backend():
                    FileOps.copy_folder(saved_folder, self.get_local_worker_path())
                self.model = model
        except Exception:
            logging.info(
                'Checkpoint file is not existed, use default model now.')
            return

    def _init_metrics(self, metrics=None):
        """Init metrics."""
        if metrics is not None:
            return metrics
        else:
            return Metrics()

    def _init_dataloader(self, mode, loader=None):
        """Init dataloader."""
        if loader is not None:
            return loader
        if mode == "train" and self.hps is not None and self.hps.get("dataset") is not None:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode, hps=self.hps.get("dataset"))
        else:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode)
        if vega.is_torch_backend():
            if self.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=hvd.size(), rank=hvd.rank())
                dataset.sampler = sampler
            return dataset.dataloader
        elif vega.is_tf_backend():
            if self.distributed:
                dataset.set_distributed(self._world_size, self._rank_id)
            return dataset

    def _train_loop(self):
        """Do the training with data, callbacks and step functions etc."""
        # Allow user to build trainer in before_train() callback, but they
        # should set lazy_built in configuration file to True
        self.callbacks.before_train()
        if self.skip_train:
            return
        for epoch in range(self.epochs):
            epoch_logs = {'train_num_batches': len(self.train_loader)}
            if self.do_validation:
                epoch_logs.update({'valid_num_batches': len(self.valid_loader)})
            self.callbacks.before_epoch(epoch, epoch_logs)
            self._train_epoch()
            if self.do_validation and self._should_run_validation(epoch):
                self._valid_epoch()
            self.callbacks.after_epoch(epoch)
        self.callbacks.after_train()
        if self.distributed:
            self._shutdown_distributed()

    def _train_epoch(self):
        if vega.is_torch_backend():
            self.model.train()
            for batch_index, batch in enumerate(self.train_loader):
                batch = self.make_batch(batch)
                batch_logs = {'train_batch': batch}
                self.callbacks.before_train_step(batch_index, batch_logs)
                train_batch_output = self.train_step(batch)
                batch_logs.update(train_batch_output)
                if self.config.is_detection_trainer:
                    batch_logs.update({'is_detection_trainer': True})
                self.callbacks.after_train_step(batch_index, batch_logs)
        elif vega.is_tf_backend():
            self.estimator.train(input_fn=self.train_input_fn,
                                 steps=len(self.train_loader),
                                 hooks=self._init_logging_hook())

    def _valid_epoch(self):
        self.callbacks.before_valid()
        valid_logs = None
        if vega.is_torch_backend():
            self.model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(self.valid_loader):
                    batch = self.make_batch(batch)
                    batch_logs = {'valid_batch': batch}
                    self.callbacks.before_valid_step(batch_index, batch_logs)
                    valid_batch_output = self.valid_step(batch)
                    self.callbacks.after_valid_step(batch_index, valid_batch_output)
        elif vega.is_tf_backend():
            eval_metrics = self.estimator.evaluate(input_fn=self.valid_input_fn,
                                                   steps=len(self.valid_loader))
            self.valid_metrics.update(eval_metrics)
            valid_logs = dict()
            valid_logs['cur_valid_perfs'] = self.valid_metrics.results
        self.callbacks.after_valid(valid_logs)

    def _init_step_functions(self, make_batch=None,
                             train_step=None, valid_step=None):
        # Init make_batch function by user or using the default one
        if self.make_batch is None:
            if make_batch is not None:
                self.make_batch = make_batch
            else:
                self.make_batch = self._default_make_batch

        # Init train_step function by user or using the default one
        if self.train_step is None:
            if train_step is not None:
                self.train_step = train_step
            else:
                self.train_step = self._default_train_step

        # Init valid_step function by user or using the default one
        if self.valid_step is None:
            if valid_step is not None:
                self.valid_step = valid_step
            else:
                self.valid_step = self._default_valid_step

    def _default_make_batch(self, batch):
        """Unpack batch to get input and target."""
        input, target = batch
        if self.use_cuda and not self.config.is_detection_trainer:
            input, target = input.cuda(), target.cuda()
        return (input, target)

    def _default_train_step(self, batch):
        input, target = batch
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, target)
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.optimizer.synchronize()
            with self.optimizer.skip_synchronize():
                self.optimizer.step()
        else:
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': output}

    def _default_valid_step(self, batch):
        input, target = batch
        if self.config.is_detection_trainer:
            output = self.model(input, forward_train=False)
        else:
            output = self.model(input)
        return {'valid_batch_output': output}

    def _init_estimator_fn(self, model_fn, train_input_fn, valid_input_fn):
        if self.model_fn is None:
            if model_fn is not None:
                self.model_fn = model_fn
            else:
                self.model_fn = self._default_model_fn

        if self.train_input_fn is None:
            if train_input_fn is not None:
                self.train_input_fn = train_input_fn
            else:
                self.train_input_fn = self._default_train_input_fn

        if self.valid_input_fn is None:
            if valid_input_fn is not None:
                self.valid_input_fn = valid_input_fn
            else:
                self.valid_input_fn = self._default_valid_input_fn

    def _init_minimize_op(self, loss, global_step, var_list=None):
        """Init loss minimize operation, include loss scale method."""
        loss_scale = self.config.loss_scale if self.use_amp else 1.
        if loss_scale != 1:
            scaled_grad_vars = self.optimizer.compute_gradients(loss * loss_scale, var_list=var_list)
            unscaled_grad_vars = [(grad / loss_scale, var) for grad, var in scaled_grad_vars]
            minimize_op = self.optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            grad_vars = self.optimizer.compute_gradients(loss, var_list=var_list)
            minimize_op = self.optimizer.apply_gradients(grad_vars, global_step)
        return minimize_op

    def _default_train_input_fn(self):
        return self.train_loader.input_fn()

    def _default_valid_input_fn(self):
        return self.valid_loader.input_fn()

    def _default_model_fn(self, features, labels, mode):
        """Define model_fn used by TensorFlow Estimator.

        :params features: input features
        :type features: tensorflow tensors
        :params labels: label data
        :type labels: tensorflow tensors
        :params mode: mode of estimator
        :type mode: tf.estimator.ModeKeys
        :return: tensorflow EstimatorSpec
        :rtype: tf.estimator.EstimatorSpec
        """
        logging.info('model function action')
        logits = self.model(features, mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.cast(logits, tf.float32)
        self.loss = Loss()()
        loss = self.loss(logits=logits, labels=labels)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.train_loader), tf.float32)
            self.lr_scheduler = LrScheduler()()
            self.optimizer = Optimizer()(lr_scheduler=self.lr_scheduler,
                                         epoch=epoch,
                                         distributed=self.distributed)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            minimize_op = self._init_minimize_op(loss, global_step)
            train_op = tf.group(minimize_op, update_ops)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.valid_metrics(logits, labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def _should_run_validation(self, epoch):
        # Zero valid_interval means doesn't run _valid_loop of the trainer
        # and user may provide _valid_loop in other callbacks
        if self.valid_interval == 0:
            return False
        else:
            return epoch % self.valid_interval == 0 or (epoch + 1) == self.epochs

    def _init_callbacks(self, callbacks):
        # Initialize custom callbacks by configuration or parameters
        if callbacks is not None:
            return callbacks
        disables = []
        if not self.config.model_statistics:
            disables.append('ModelStatistics')
        self.callbacks = CallbackList(self.config.callbacks, disables)
        self.callbacks.set_trainer(self)

    def _metric_average(self, val, name):
        """Do metric average.

        :param val: input value
        :param name: metric name
        :return:
        """
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    @property
    def _first_rank(self):
        """Check if the first rank."""
        if self.distributed and hvd.rank() != 0:
            return False
        else:
            return True

    def _backup(self):
        """Backup result worker folder."""
        if self.need_backup is True and self.backup_base_path is not None:
            backup_worker_path = FileOps.join_path(
                self.backup_base_path, self.get_worker_subpath())
            FileOps.copy_folder(
                self.get_local_worker_path(self.step_name, self.worker_id), backup_worker_path)

    def _save_visual_data(self, is_train=True, pfms=None, loss=None, lr=None):
        # TODO Will move to metric base class later.
        for _name, value in pfms.items():
            if is_train:
                _name = "{}_{}".format("t", _name)
            else:
                _name = "{}_{}".format("v", _name)
            if isinstance(value, list):
                for i, _item in enumerate(value):
                    _name = "{}_{}".format(_name, i)
                    self.visual_data[_name] = _item.data.item()
            elif isinstance(value, dict):
                for k, v in value.keys():
                    _name = "{}_{}".format(k, i)
                    self.visual_data[_name] = v
            elif value is not None:
                self.visual_data[_name] = value.data.item()
        if loss is not None:
            self.visual_data["loss"] = loss
        if lr is not None:
            self.visual_data["lr"] = lr

    def _init_tf_estimator(self):
        """Init tensorflow estimator."""
        if not vega.is_tf_backend():
            return
        sess_config = self._init_session_config()
        if vega.is_gpu_device():
            self._init_gpu_estimator(sess_config)
        elif vega.is_npu_device():
            self._init_npu_estimator(sess_config)

    def _init_tf_session(self):
        if not vega.is_tf_backend():
            return
        sess_config = self._init_session_config()
        self.sess = tf.Session(config=sess_config)

    def _init_session_config(self):
        sess_config = self._init_gpu_session_config() if vega.is_gpu_device() else \
            self._init_npu_session_config()
        return sess_config

    def _init_logging_hook(self):
        logging_hook = []
        if vega.is_gpu_device() and self.distributed:
            logging_hook += [hvd.BroadcastGlobalVariablesHook(0)]
        return logging_hook

    def _init_gpu_estimator(self, sess_config):
        """Init tensorflow estimator."""
        config = tf.estimator.RunConfig(model_dir=self.get_local_worker_path(),
                                        save_checkpoints_steps=self.config.save_steps,
                                        log_step_count_steps=self.config.report_freq,
                                        session_config=sess_config)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                config=config)

    def _init_npu_estimator(self, sess_config):
        model_dir = self.get_local_worker_path()
        if self.distributed:
            model_dir = FileOps.join_path(model_dir, str(self._local_rank_id))
        config = NPURunConfig(model_dir=model_dir,
                              save_checkpoints_steps=self.config.save_steps,
                              log_step_count_steps=self.config.report_freq,
                              session_config=sess_config,
                              enable_data_pre_proc=True,
                              iterations_per_loop=1)
        self.estimator = NPUEstimator(model_fn=self.model_fn,
                                      config=config)

    def _init_gpu_session_config(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        if self.distributed:
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        return sess_config

    def _init_npu_session_config(self):
        sess_config = tf.ConfigProto()
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        if self.use_amp:
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["use_off_line"].b = True
        # custom_op.parameter_map['hcom_parallel'].b = True
        # custom_op.parameter_map["enable_data_pre_proc"].b = True
        # custom_op.parameter_map["mix_compile_mode"].b = True  # mixed calculation
        # custom_op.parameter_map["min_group_size"].b = 1
        return sess_config

    def _shutdown_distributed(self):
        if vega.is_npu_device() and self.distributed:
            self.sess.run(self.npu_shutdown)
            self.sess.close()
