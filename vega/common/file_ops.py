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

"""FileOps class."""
import os
import pickle
import logging
import shutil

logger = logging.getLogger(__name__)


class FileOps(object):
    """This is a class with some class methods to handle some files or folder."""

    @classmethod
    def make_dir(cls, *args):
        """Make new a local directory.

        :param * args: list of str path to joined as a new directory to make.
        :type * args: list of str args

        """
        _path = cls.join_path(*args)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def make_base_dir(cls, *args):
        """Make new a base directory.

        :param * args: list of str path to joined as a new base directory to make.
        :type * args: list of str args

        """
        _file = cls.join_path(*args)
        if os.path.isfile(_file):
            return
        _path, _ = os.path.split(_file)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def join_path(cls, *args):
        """Join list of path and return.

        :param * args: list of str path to be joined.
        :type * args: list of str args
        :return: joined path str.
        :rtype: str

        """
        if len(args) == 1:
            return args[0]
        args = list(args)
        for i in range(1, len(args)):
            if args[i][0] in ["/", "\\"]:
                args[i] = args[i][1:]
        # local path
        if ":" not in args[0]:
            args = tuple(args)
            return os.path.join(*args)
        # http or s3 path
        prefix = args[0]
        if prefix[-1] != "/":
            prefix += "/"
        tail = os.path.join(*args[1:])
        return prefix + tail

    @classmethod
    def dump_pickle(cls, obj, filename, protocol=None):
        """Dump a object to a file using pickle.

        :param object obj: target object.
        :param str filename: target pickle file path.

        """
        if not os.path.isfile(filename):
            cls.make_base_dir(filename)
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=protocol)

    @classmethod
    def load_pickle(cls, filename, fix_imports=True, encoding="ASCII", errors="strict"):
        """Load a pickle file and return the object.

        :param str filename: target pickle file path.
        :return: return the loaded original object.
        :rtype: object or None.

        """
        if not os.path.isfile(filename):
            return None
        from vega.common.general import General
        from vega.security.load_pickle import restricted_loads
        with open(filename, "rb") as f:
            return restricted_loads(
                f, fix_imports=fix_imports, encoding=encoding, errors=errors, security=General.security)

    @classmethod
    def copy_folder(cls, src, dst):
        """Copy a folder from source to destination.

        :param str src: source path.
        :param str dst: destination path.

        """
        if dst is None or dst == "":
            return
        try:
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                else:
                    if not os.path.samefile(src, dst):
                        for files in os.listdir(src):
                            name = os.path.join(src, files)
                            back_name = os.path.join(dst, files)
                            if os.path.isfile(name):
                                shutil.copy(name, back_name)
                            else:
                                if not os.path.isdir(back_name):
                                    shutil.copytree(name, back_name)
                                else:
                                    cls.copy_folder(name, back_name)
            else:
                logger.error("failed to copy folder, folder is not existed, folder={}.".format(src))
        except Exception as ex:
            logger.error("failed to copy folder, src={}, dst={}, msg={}".format(src, dst, str(ex)))

    @classmethod
    def copy_file(cls, src, dst):
        """Copy a file from source to destination.

        :param str src: source path.
        :param str dst: destination path.

        """
        if dst is None or dst == "":
            return
        try:
            if os.path.isfile(src):
                shutil.copy(src, dst)
            else:
                logger.error(f"failed to copy file, file is not existed, file={src}.")
        except OSError as os_error:
            if os_error.errno == 13 and os.path.abspath(os_error.filename) != os.path.abspath(src):
                need_try_again = True
                os_error_filename = os.path.abspath(os_error.filename)
            else:
                logger.error(f"Failed to copy file, src={src}, dst={dst}, msg={os_error}")
        except Exception as ex:
            logger.error(f"Failed to copy file, src={src}, dst={dst}, msg={ex}")

        if "need_try_again" in locals():
            try:
                logger.info("The dest file is readonly, remove the dest file and try again.")
                os.remove(os_error_filename)
                shutil.copy(src, dst)
            except Exception as ex:
                logger.error(f"Failed to copy file after removed dest file, src={src}, dst={dst}, msg={ex}")

    @classmethod
    def download_dataset(cls, src_path, local_path=None):
        """Download dataset from http or https web site, return path.

        :param src_path: the data path
        :type src_path: str
        :param local_path: the local path
        :type local_path: str
        :raises FileNotFoundError: if the file path is not exist, an error will raise
        :return: the final data path
        :rtype: str
        """
        if src_path is None:
            raise FileNotFoundError("Dataset path is None, please set dataset path in config file.")
        if os.path.exists(src_path):
            return src_path
        else:
            raise FileNotFoundError('Path is not existed, path={}'.format(src_path))

    @classmethod
    def download_pretrained_model(cls, src_path):
        """Download dataset from http or https web site, return path.

        :param src_path: the data path
        :type src_path: str
        :raises FileNotFoundError: if the file path is not exist, an error will raise
        :return: the final data path
        :rtype: str
        """
        if src_path is None:
            raise FileNotFoundError("Path of pretrained model is None, please set correct path.")
        if os.path.isfile(src_path):
            return src_path
        else:
            raise FileNotFoundError('Model is not existed, path={}'.format(src_path))

    @classmethod
    def _untar(cls, src, dst=None):
        import tarfile
        if dst is None:
            dst = os.path.dirname(src)
        with tarfile.open(src, 'r:gz') as tar:
            tar.extractall(path=dst)

    @classmethod
    def exists(cls, path):
        """Is folder existed or not.

        :param folder: folder
        :type folder: str
        :return: folder existed or not.
        :rtype: bool
        """
        return os.path.isdir(path) or os.path.isfile(path)

    @classmethod
    def remove(cls, path):
        """Remove file."""
        if not os.path.exists(path):
            return
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception:
            logger.warn(f"Failed to remove file/dir: {path}")
