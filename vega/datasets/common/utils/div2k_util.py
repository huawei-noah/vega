# -*- coding: utf-8 -*-

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

"""This script contains some common function to process the DIV2K dataset."""

import os
import glob
import cv2
import numpy as np
from vega.common import FileOps

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
PKL_EXTENSIONS = ['.pkl', '.pt', '.pth']


def is_image_file(filename):
    """To judeg whether a given file name is a image or not.

    :param filename: the input filename
    :type filename: str
    :return: true or false
    :rtype: bool
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# image/pkl files will always in the same folder; otherwise use subfile
def get_paths_from_dir(path):
    """Get all the files in the given directory.

    :param path: the given directory
    :type path: str
    :return: the file name and its path
    :rtype: list
    """
    return sorted(glob.glob(os.path.join(path, "*")))


def get_files_datatype(file_names):
    """Get the datatype of the file.

    :param file_names: input file name
    :type file_names: str
    :raises NotImplementedError: if the datatype in not in 'img' and 'pkl'
    :return: the datatype of the file
    :rtype: str
    """
    extensions = {'.' + file_name.split('.')[-1] for file_name in file_names}
    if extensions.issubset(set(IMG_EXTENSIONS)):
        return 'img'
    elif extensions.issubset(set(PKL_EXTENSIONS)):
        return 'pkl'
    else:
        raise NotImplementedError('Datatype not recognized!')


def get_datatype(dataroot):
    """Get the datatype of the data patrh.

    :param dataroot: the data path
    :type dataroot: str
    :return: the datatype
    :rtype: str
    """
    if dataroot.endswith(".lmdb"):
        return "lmdb"
    file_names = os.listdir(dataroot)
    return get_files_datatype(file_names)


def read_img_pkl(path):
    """Real image from a pkl file.

    :param path: the file path
    :type path: str
    :return: the image
    :rtype: tuple
    """
    return FileOps.load_pickle(path)


def read_img_img(path):
    """Read the picture format image.

    :param path: the image path
    :type path: str
    :return: the image data
    :rtype: ndarray
    """
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def np_to_tensor(np_array):
    """Convert an image from np array to tensor.

    :param np_array: image in np.array, np.uint8, HWC, BGR
    :type np_array: np.array
    :return: image tensor, torch.float32, CHW, BGR
    :rtype: tensor
    """
    np_array = np.asarray(np_array)
    return np.transpose(np_array, (2, 0, 1)).astype(np.float32)
