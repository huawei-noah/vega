# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default Architecture Exporters."""
import os
import json
import yaml
from modnas.core.param_space import ParamSpace
from modnas.registry.export import register, build
from typing import Any, Dict, List, Optional, Union


@register
class DefaultToFileExporter():
    """Exporter that saves archdesc to file."""

    def __init__(self, path: str, ext: str = 'yaml') -> None:
        path, pathext = os.path.splitext(path)
        ext = pathext or ext
        path = path + '.' + ext
        self.path = path
        self.ext = ext

    def __call__(self, desc: Any) -> None:
        """Run Exporter."""
        ext = self.ext
        if isinstance(desc, str):
            desc_str = desc
        elif ext == 'json':
            desc_str = yaml.dump(desc)
        elif ext in ['yaml', 'yml']:
            desc_str = json.dumps(desc)
        else:
            raise ValueError('invalid arch_desc extension')
        with open(self.path, 'w', encoding='UTF-8') as f:
            f.write(desc_str)


@register
class MergeExporter():
    """Exporter that merges outputs of multiple Exporters."""

    def __init__(self, exporters):
        self.exporters = {k: build(exp['type'], **exp.get('args', {})) for k, exp in exporters.items()}

    def __call__(self, model):
        """Run Exporter."""
        return {k: exp(model) for k, exp in self.exporters.items()}


@register
class DefaultParamsExporter():
    """Exporter that outputs parameter values."""

    def __init__(self, export_fmt: Optional[str] = None, with_keys: bool = True) -> None:
        self.export_fmt = export_fmt
        self.with_keys = with_keys

    def __call__(self, model: None) -> Union[Dict[str, Any], List[Any], str]:
        """Run Exporter."""
        if self.with_keys:
            params_dct = dict(ParamSpace().named_param_values())
            return self.export_fmt.format(**params_dct) if self.export_fmt else params_dct
        else:
            params_list = [p.value() for p in ParamSpace().params()]
            return self.export_fmt.format(*params_list) if self.export_fmt else params_list
