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

"""Trainer statistics reporter."""
from functools import partial
from modnas.registry.callback import register
from modnas.utils import format_dict, AverageMeter
from ..base import CallbackBase


@register
class TrainerReporter(CallbackBase):
    """Trainer statistics reporter class."""

    priority = -1

    def __init__(self, interval=0.2, format_fn=None, stat_cls=None):
        super().__init__({
            'after:TrainerBase.train_step': partial(self.report_step, 'train'),
            'after:TrainerBase.valid_step': partial(self.report_step, 'valid'),
            'after:TrainerBase.train_epoch': partial(self.report_epoch, 'train'),
            'after:TrainerBase.valid_epoch': partial(self.report_epoch, 'valid'),
            'after:TrainerBase.loss': self.on_loss,
        })
        self.interval = interval
        self.format_fn = format_fn
        self.last_batch_size = 1
        self.stat_cls = stat_cls or AverageMeter
        self.stats = {}

    def init_stats(self, proc, keys):
        """Initialize statistics."""
        self.stats[proc] = {k: self.stat_cls() for k in keys}

    def reset(self):
        """Reset statistics."""
        self.stats.clear()
        self.last_batch_size = 1

    def on_loss(self, ret, trainer, output, data, model):
        """Record batch size in each loss call."""
        self.last_batch_size = len(data[-1])

    def report_epoch(self, proc, ret, *args, **kwargs):
        """Log statistics report in each epoch."""
        ret = ret or {}
        proc_stats = self.stats.get(proc)
        if proc_stats and not ret:
            ret.update({k: v.avg for k, v in proc_stats.items()})
        self.reset()
        return None if not ret else ret

    def report_step(self, proc, ret, trainer, estim, model, epoch, tot_epochs, step, tot_steps):
        """Log statistics report in each step."""
        if step >= tot_steps:
            return
        if step == 0:
            self.reset()
        cur_step = epoch * tot_steps + step
        interval = self.interval
        if interval and interval < 1:
            interval = int(interval * tot_steps)
        stats = ret.copy() if isinstance(ret, dict) else {}
        stats = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
        stats_len = stats.pop('N', self.last_batch_size)
        if proc not in self.stats and stats:
            self.init_stats(proc, stats.keys())
        proc_stats = self.stats[proc]
        writer = trainer.writer
        for k, v in stats.items():
            proc_stats[k].update(v, n=stats_len)
            if writer is not None:
                writer.add_scalar('/'.join(['trainer', proc, k]), v, cur_step)
        if interval is None or (interval != 0 and (step + 1) % interval == 0) or step + 1 == tot_steps:
            fmt_info = format_dict({k: v.avg for k, v in proc_stats.items()}, fmt_val=self.format_fn)
            trainer.logger.info('{}: [{:3d}/{}] {}'.format(proc.title(), step + 1, tot_steps, fmt_info))
