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

"""Default Torch Exporters."""
import logging
import traceback
import torch
from modnas.registry.export import register
from modnas.utils.logging import get_logger


logger = get_logger('export')


@register
class DefaultTorchCheckpointExporter():
    """Exporter that saves model checkpoint to file."""

    def __init__(self, path, zip_file=None):
        self.path = path
        save_kwargs = {}
        if zip_file is not None and int('.'.join(torch.__version__.split('.'))) >= 140:
            save_kwargs['_use_new_zipfile_serialization'] = zip_file
        self.save_kwargs = save_kwargs

    def __call__(self, model):
        """Run Exporter."""
        logger.info('Saving torch checkpoint to {}'.format(self.path))
        try:
            torch.save(model.state_dict(), self.path, **self.save_kwargs)
        except RuntimeError as e:
            logger.debug(traceback.format_exc())
            logger.error(f'Failed saving checkpoint, message: {e}')
        return model
