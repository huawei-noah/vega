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

"""Import hooks for ModularNAS (PEP 302)."""

import sys
import importlib
import importlib.util


class ModNASImporter():
    """ModularNAS Importer class."""

    path_exclude = [
        'modnas.registry'
    ]

    path_spec = [
        ('modnas.contrib.arch_space', 'vega.networks.pytorch.customs.modnas.contrib.arch_space'),
        ('modnas.arch_space', 'vega.networks.pytorch.customs.modnas.arch_space'),
        ('modnas', 'vega.algorithms.nas.modnas'),
    ]

    def find_spec(self, fullname, path, target=None):
        """Handle ModularNAS imports."""
        for exc_path in self.path_exclude:
            if exc_path in fullname:
                return
        for ori_path, _ in self.path_spec:
            if fullname.startswith(ori_path):
                return importlib.util.spec_from_loader(fullname, self)

    def load_module(self, fullname):
        """Load ModularNAS module by import path."""
        for ori_path, cur_path in self.path_spec:
            if not fullname.startswith(ori_path):
                continue
            cur_fullname = fullname.replace(ori_path, cur_path)
            mod = sys.modules.get(fullname, sys.modules.get(cur_fullname, importlib.import_module(cur_fullname)))
            mod.__package__ = fullname
            mod.__name__ = fullname
            sys.modules[fullname] = mod
            return mod


sys.meta_path.append(ModNASImporter())
