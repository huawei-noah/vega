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
from vega.core.common.class_factory import ClassFactory
from vega.core.common.user_config import UserConfig
from vega.core.common.file_ops import FileOps

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
except:
    pass
FileOps.copy_file(args.cf_file, './cf_file.pickle')
hvd.join()
with open('./cf_file.pickle', 'rb') as f:
    cf_content = pickle.load(f)
ClassFactory.__configs__ = cf_content.get('configs')
ClassFactory.__registry__ = cf_content.get('registry')
UserConfig().__data__ = cf_content.get('data')
cls_trainer = ClassFactory.get_cls('trainer')
trainer = cls_trainer(None, 0)
trainer.train_process()
