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

"""Load pickle."""

import pickle

__all__ = ["restricted_loads"]


safe_builtins = {
    'vega',
    'torch',
    'torchvision',
    'functools',
    'timm',
    'mindspore',
    'tensorflow',
    'numpy',
    'imageio',
    'collections',
    'apex',
    'ascend_automl'
}


class RestrictedUnpickler(pickle.Unpickler):
    """Restrict unpickler."""

    def __init__(self, file, fix_imports, encoding, errors, security):
        super(RestrictedUnpickler, self).__init__(file=file, fix_imports=fix_imports, encoding=encoding, errors=errors)
        self.security = security

    def find_class(self, module, name):
        """Find class."""
        _class = super().find_class(module, name)
        if self.security:
            if module.split('.')[0] in safe_builtins:
                return _class
            raise pickle.UnpicklingError(f"global '{module}' is forbidden")
        else:
            return _class


def restricted_loads(file, fix_imports=True, encoding="ASCII", errors="strict", security=False):
    """Load obj."""
    return RestrictedUnpickler(file, fix_imports=fix_imports, encoding=encoding, errors=errors,
                               security=security).load()
