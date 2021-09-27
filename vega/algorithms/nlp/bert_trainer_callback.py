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
from vega.common import ClassFactory, ClassType
from vega.trainer.trainer_ms import TrainerMs
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model as MsModel
from mindspore.train.train_thor import ConvertModelUtils
from mindspore import context
from mindspore.nn.optim import Lamb, Momentum, AdamWeightDecay, thor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.metrics import Metric
from .src import BertNetworkWithLoss, BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell, \
    BertTrainAccumulationAllReduceEachWithLossScaleCell, \
    BertTrainAccumulationAllReducePostWithLossScaleCell, \
    BertTrainOneStepWithLossScaleCellForAdam, \
    AdamWeightDecayForBert, AdamWeightDecayOp
from .src.dataset import create_bert_dataset
from .src.utils import LossCallBack, BertLearningRate
from .src import BertModel, GetMaskedLMOutput

logger = logging.getLogger(__name__)


class myMetric(Metric):
    """Self-defined Metric as a callback."""

    def __init__(self):
        super(myMetric, self).__init__()
        self.clear()

    def clear(self):
        """Construct the trainer of Bert."""
        self.total_num = 0
        self.acc_num = 0

    def update(self, *inputs):
        """Construct the trainer of Bert."""
        total_num = self._convert_data(inputs[0])
        acc_num = self._convert_data(inputs[1])
        self.total_num = total_num
        self.acc_num = acc_num

    def eval(self):
        """Construct the trainer of Bert."""
        return self.acc_num / self.total_num


class GetLogProbs(nn.Cell):
    """Get MaskedLM prediction scores."""

    def __init__(self, config):
        super(GetLogProbs, self).__init__()
        self.bert = BertModel(config, False)
        self.cls1 = GetMaskedLMOutput(config)

    def construct(self, input_ids, input_mask, token_type_id, masked_pos):
        """Construct the trainer of Bert."""
        sequence_output, _, embedding_table = self.bert(input_ids, token_type_id, input_mask)
        prediction_scores = self.cls1(sequence_output, embedding_table, masked_pos)
        return prediction_scores


class BertPretrainEva(nn.Cell):
    """Evaluate MaskedLM prediction scores."""

    def __init__(self, config):
        super(BertPretrainEva, self).__init__()
        self.bert = GetLogProbs(config)
        self.argmax = P.Argmax(axis=-1, output_type=mstype.int32)
        self.equal = P.Equal()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.total = Parameter(Tensor([0], mstype.float32))
        self.acc = Parameter(Tensor([0], mstype.float32))
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask, token_type_id, masked_pos, masked_ids, masked_weights, nsp_label):
        """Calculate prediction scores."""
        bs, _ = self.shape(input_ids)
        probs = self.bert(input_ids, input_mask, token_type_id, masked_pos)
        index = self.argmax(probs)
        index = self.reshape(index, (bs, -1))
        eval_acc = self.equal(index, masked_ids)
        eval_acc1 = self.cast(eval_acc, mstype.float32)
        real_acc = eval_acc1 * masked_weights
        acc = self.sum(real_acc)
        total = self.sum(masked_weights)
        self.total += total
        self.acc += acc
        return acc, self.total, self.acc


def get_enwiki_512_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    """Get enwiki dataset when seq_length is 512."""
    from .src.model_utils.config import config as cfg, bert_net_cfg
    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask", "segment_ids",
                                                                            "masked_lm_positions", "masked_lm_ids",
                                                                            "masked_lm_weights",
                                                                            "next_sentence_labels"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="segment_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="masked_lm_ids")
    ds = ds.map(operations=type_cast_op, input_columns="masked_lm_positions")
    ds = ds.map(operations=type_cast_op, input_columns="next_sentence_labels")
    ds = ds.repeat(repeat_count)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def bert_predict():
    """Predict function."""
    from .src.model_utils.config import config as cfg, bert_net_cfg
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    dataset = get_enwiki_512_dataset(cfg.batch_size, 1)
    net_for_pretraining = BertPretrainEva(bert_net_cfg)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = MsModel(net_for_pretraining)
    return model, dataset, net_for_pretraining


def _get_optimizer(args_opt, network):
    """Get bert optimizer, support Lamb, Momentum, AdamWeightDecay."""
    from .src.model_utils.config import config as cfg, bert_net_cfg
    if cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=cfg.Lamb.learning_rate,
                                       end_learning_rate=cfg.Lamb.end_learning_rate,
                                       warmup_steps=cfg.Lamb.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.Lamb.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.Lamb.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.Lamb.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.Lamb.weight_decay},
                        {'params': other_params},
                        {'order_params': params}]
        optimizer = Lamb(group_params, learning_rate=lr_schedule, eps=cfg.Lamb.eps)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=cfg.Momentum.learning_rate,
                             momentum=cfg.Momentum.momentum)
    elif cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=cfg.AdamWeightDecay.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': params}]
        if args_opt.enable_lossscale == "true" and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayForBert(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        elif context.get_context("mode") == context.PYNATIVE_MODE and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayOp(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        else:
            optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
    elif cfg.optimizer == "Thor":
        from .src.utils import get_bert_thor_lr, get_bert_thor_damping
        lr = get_bert_thor_lr(cfg.Thor.lr_max, cfg.Thor.lr_min, cfg.Thor.lr_power, cfg.Thor.lr_total_steps)
        damping = get_bert_thor_damping(cfg.Thor.damping_max, cfg.Thor.damping_min, cfg.Thor.damping_power,
                                        cfg.Thor.damping_total_steps)
        split_indices = None
        if bert_net_cfg.num_hidden_layers == 12 and not bert_net_cfg.use_relative_positions:
            split_indices = [28, 55, 77]
        elif bert_net_cfg.num_hidden_layers == 24 and not bert_net_cfg.use_relative_positions:
            split_indices = [38, 93, 149]
        optimizer = thor(network, lr, damping, cfg.Thor.momentum,
                         cfg.Thor.weight_decay, cfg.Thor.loss_scale, cfg.batch_size,
                         decay_filter=lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
                         split_indices=split_indices, enable_clip_grad=True, frequency=cfg.Thor.frequency)
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, AdamWeightDecay, Thor]".
                         format(cfg.optimizer))
    return optimizer


@ClassFactory.register(ClassType.TRAINER)
class BertTrainerCallback(TrainerMs):
    """Construct the trainer of Bert."""

    disable_callbacks = ['ProgressLogger']

    def build(self):
        """Construct the trainer of Bert."""
        logging.debug("Trainer Config: {}".format(self.config))
        self._init_hps()
        self.do_validation = False
        self.use_syncbn = self.config.syncbn
        if not self.train_loader:
            self.train_loader = create_bert_dataset(int(os.environ.get("RANK_SIZE", "1")),
                                                    int(os.environ.get("RANK_ID", "0")), True,
                                                    '/root/lzc/zhwiki/wikidata/new/', '', 32)
        if not self.valid_loader:
            self.valid_loader = create_bert_dataset(int(os.environ.get("RANK_SIZE", "1")),
                                                    int(os.environ.get("RANK_ID", "0")), True,
                                                    '/root/lzc/zhwiki/wikidata/new/', '', 32)
        self.batch_num_train = self.train_loader.get_dataset_size()
        self.batch_num_valid = self.valid_loader.get_dataset_size()

    def _train_epoch(self):
        """Construct the trainer of Bert."""
        from .src.model_utils.config import config as cfg, bert_net_cfg
        cfg.train_steps = cfg.epoch_size * self.train_loader.get_dataset_size() // cfg.accumulation_steps
        optimizer = _get_optimizer(cfg, self.model)

        if cfg.enable_lossscale == "true":
            update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                     scale_factor=cfg.scale_factor,
                                                     scale_window=cfg.scale_window)
            accumulation_steps = cfg.accumulation_steps
            enable_global_norm = cfg.enable_global_norm
            if accumulation_steps <= 1:
                if cfg.optimizer == 'AdamWeightDecay' and cfg.device_target == 'GPU':
                    net_with_grads = BertTrainOneStepWithLossScaleCellForAdam(self.model, optimizer=optimizer,
                                                                              scale_update_cell=update_cell)
                else:
                    net_with_grads = BertTrainOneStepWithLossScaleCell(self.model, optimizer=optimizer,
                                                                       scale_update_cell=update_cell)
            else:
                allreduce_post = cfg.distribute == "false" or cfg.allreduce_post_accumulation == "true"
                net_with_accumulation = (BertTrainAccumulationAllReducePostWithLossScaleCell if allreduce_post else
                                         BertTrainAccumulationAllReduceEachWithLossScaleCell)
                net_with_grads = net_with_accumulation(self.model, optimizer=optimizer,
                                                       scale_update_cell=update_cell,
                                                       accumulation_steps=accumulation_steps,
                                                       enable_global_norm=enable_global_norm)
        else:
            net_with_grads = BertTrainOneStepCell(self.model, optimizer=optimizer, enable_clip_grad=True)
            if cfg.optimizer == "Thor":
                net_with_grads = BertTrainOneStepCell(self.model, optimizer=optimizer, sens=cfg.Thor.loss_scale,
                                                      enable_clip_grad=False)

        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps, keep_checkpoint_max=1)
        save_path = self.get_local_worker_path(self.step_name, self.worker_id)
        ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
        loss_cb = LossMonitor()
        callback_list = [ckpoint_cb, loss_cb]
        model = MsModel(net_with_grads)
        self.ms_model = ConvertModelUtils().convert_to_thor_model(model, network=net_with_grads, optimizer=optimizer)
        try:
            self.ms_model.train(epoch=self.epochs,
                                train_dataset=self.train_loader,
                                callbacks=callback_list,
                                dataset_sink_mode=False)
        except RuntimeError as e:
            logging.warning(f"failed to train the model, skip it, message: {str(e)}")

    def _valid_epoch(self):
        """Construct the trainer of Bert."""
        _, dataset, net_for_pretraining = bert_predict()
        net = MsModel(net_for_pretraining, eval_network=net_for_pretraining, eval_indexes=[0, 1, 2],
                      metrics={'name': myMetric()})
        res = net.eval(dataset, dataset_sink_mode=False)
        logging.info('Accuracy is: {}'.format(res))
        valid_logs = dict()
        valid_logs['cur_valid_perfs'] = res
        self.callbacks.after_valid(valid_logs)
