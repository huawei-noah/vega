# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Register factory."""

import importlib
import os
from os.path import dirname
import sys
import glob
from absl import logging
# logging.set_verbosity(logging.DEBUG)
try:
    import xt
except ModuleNotFoundError:
    xt = None
from zeus import set_backend
set_backend(backend='tensorflow', device_category='GPU')


class RegisterStub(object):  # pylint: disable=too-few-public-methods
    """Register module."""

    def __init__(self, name):
        """Initialize Register stub."""
        self._dict = dict()
        self._name = name

    def __getitem__(self, key):
        """Get item."""
        try:
            return self._dict[key]
        except Exception as exc:
            logging.error("module {} not found: {}".format(key, exc))
            raise exc

    def __call__(self, param):
        """Call function."""
        if not callable(param):
            raise Exception("To Registry must be callable, Got: {}.".format(param))

        register_name = param.__name__
        if register_name in self._dict:
            logging.warning("Key:{} is registered, will replace with {}.".format(
                register_name, self._name))

        self._dict[register_name] = param
        return param


class Registers(object):  # pylint: disable=too-few-public-methods, invalid-name
    """All module registers within class instance."""

    def __init__(self):
        raise RuntimeError("Registries prohibit instancing !")

    agent = RegisterStub("agent")
    model = RegisterStub("model")
    algorithm = RegisterStub("algorithm")
    env = RegisterStub("env")
    comm = RegisterStub("comm")


def path_to_module_format(py_path):
    """Transform a python/file/path to module format match to the importlib."""
    # return py_path.replace("/", ".").rstrip(".py")
    return os.path.splitext(py_path)[0].replace("/", ".")


def _keep_patten(file_name):
    return False if "__init__" in file_name or "_conf" in file_name else True


def _catch_subdir_modules(module_path):
    if xt:
        work_path = os.path.join(dirname(dirname(xt.__file__)), module_path)
    else:
        work_path = module_path
    target_file = glob.glob("{}/*/*.py".format(work_path))
    model_path_len = len(work_path)
    target_file_clip = [item[model_path_len:] for item in target_file]
    used_file = [item for item in target_file_clip if _keep_patten(item)]

    return [path_to_module_format(_item) for _item in used_file]


def _catch_peer_modules(module_path):
    if xt:
        work_path = os.path.join(dirname(dirname(xt.__file__)), module_path)
        if not os.path.exists(work_path):
            work_path = os.path.join(dirname(dirname(xt.__file__)),
                                     module_path.split("/", 1)[-1])
    else:
        work_path = module_path

    target_file = glob.glob("{}/*.py".format(work_path))
    model_path_len = len(work_path)
    target_file_clip = [item[model_path_len:] for item in target_file]

    used_file = [item for item in target_file_clip if _keep_patten(item)]

    return [path_to_module_format(_item) for _item in used_file]


MODEL_MODULES = _catch_subdir_modules("xt/model")
ALG_MODULES = _catch_subdir_modules("xt/algorithm")
AGENT_MODULES = _catch_subdir_modules("xt/agent")
ENV_MODULES = _catch_subdir_modules("xt/environment")

IPC_MODULES = _catch_peer_modules("../zeus/common/ipc")

DEFAULT_MODULE_STUBS = [
    ("xt.model", MODEL_MODULES),
    ("xt.algorithm", ALG_MODULES),
    ("xt.agent", AGENT_MODULES),
    ("xt.environment", ENV_MODULES),
    ("zeus.common.ipc", IPC_MODULES),
]


def register_xt_defaults():
    """Register default modules."""
    # add `pwd` into path.
    current_work_dir = os.getcwd()
    if current_work_dir not in sys.path:
        sys.path.append(current_work_dir)

    default_modules = DEFAULT_MODULE_STUBS

    logging.debug("start to import all modules: {}".format(default_modules))
    import_error_track = list()

    for base_path, module_list in default_modules:
        for module_name in module_list:
            try:
                if base_path != "":
                    full_name = base_path + module_name
                else:
                    full_name = module_name
                importlib.import_module(full_name)
                logging.debug("Loaded {}!".format(full_name))
            except ImportError as error:
                import_error_track.append((module_name, error))

    # logging out the error among register operation
    for module, error in import_error_track:
        logging.warning("{} import failed with error:{}".format(module, error))
