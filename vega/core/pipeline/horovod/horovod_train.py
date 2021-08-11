# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Running Horovod Train."""

import os
import pickle
import argparse
import logging
import horovod.torch as hvd
from vega.common import ClassFactory
from vega.common.general import General
from vega.core.pipeline.conf import PipeStepConfig

parser = argparse.ArgumentParser(description='Horovod Fully Train')
parser.add_argument('--cf_file', type=str, help='ClassFactory pickle file')
args = parser.parse_args()

if 'VEGA_INIT_ENV' in os.environ:
    exec(os.environ.copy()['VEGA_INIT_ENV'])
logging.info('start horovod setting')
hvd.init()
try:
    import moxing as mox
    mox.file.set_auth(obs_client_log=False)
except Exception:
    pass
hvd.join()
with open(args.cf_file, 'rb') as f:
    cf_content = pickle.load(f)
model_desc = cf_content.get('model_desc')
worker_id = cf_content.get('worker_id')
ClassFactory.__registry__ = cf_content.get('registry')
General.from_dict(cf_content.get('general_config'))
PipeStepConfig.from_dict(cf_content.get('pipe_step_config'))
cls_trainer = ClassFactory.get_cls('trainer', "Trainer")
# for record in records:
trainer = cls_trainer(model_desc=model_desc, id=worker_id)
trainer.train_process()
