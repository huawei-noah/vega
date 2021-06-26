# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default Torch Exporters."""
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
        except RuntimeError:
            logger.error('Failed saving checkpoint: {}'.format(traceback.format_exc()))
        return model
