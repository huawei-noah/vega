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
"""Utils for helping user work within Huawei cloud environment."""

import os

try:
    import moxing as mox
    MA_PATH = "/home/ma-user"
    HWC_CACHE = "/cache"  # the cache capacity in hw cloud is enough.
    XT_HWC_WORKSPACE = os.path.join(HWC_CACHE, "xt_workspace")
    if not os.path.exists(XT_HWC_WORKSPACE):
        os.makedirs(XT_HWC_WORKSPACE)
except (ModuleNotFoundError, ImportError) as err:
    mox = None
    MA_PATH, XT_HWC_WORKSPACE, HWC_CACHE = None, None, None


def mox_makedir_if_not_existed(s3_path):
    """Make direction if not existed within s3."""
    check_dir = s3_path if mox.file.is_directory(s3_path) else os.path.dirname(s3_path)
    if not mox.file.is_directory(check_dir):
        mox.file.make_dirs(check_dir)


def local_makedir_if_not_existed(path):
    """Make direction if not existed within local machine."""
    check_path = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.exists(check_path):
        os.makedirs(check_path)


def sync_data_to_s3(source_data, s3_path):
    """Sync data from local machine to user's s3 path, auto-check the path firstly."""
    mox_makedir_if_not_existed(s3_path)
    if os.path.isfile(source_data):
        mox.file.copy(source_data, s3_path)
    else:  # copy dir
        mox.file.copy_parallel(source_data, s3_path)


def sync_data_from_s3(s3_path, destination):
    """Sync data from user's s3 path to local machine, auto-check the local path firstly."""
    local_makedir_if_not_existed(destination)
    if not mox.file.is_directory(s3_path):  # single s3 file
        mox.file.copy(s3_path, destination)
    else:
        mox.file.copy_parallel(s3_path, destination)
