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

"""This script is used to process the auto lane dataset."""

import os
import json
import PIL
import cv2
import numpy as np


def hwc2chw(img):
    """Transform image from HWC to CHW format.

    :param img: image to transform.
    :type: ndarray
    :return: transformed image
    :rtype: ndarray
    """
    return np.transpose(img, (2, 0, 1))


def resize_by_wh(*, img, width, height):
    """Resize image by weight and height.

    :param img:image array
    :type: ndarray
    :param width:
    :type: int
    :param height:
    :type: int
    :return:resized image
    :rtype:ndarray
    """
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def exif_transpose(img):
    """If an image has an Exif Orientation tag, transpose the image  accordingly.

    Note: Very recent versions of Pillow have an internal version
    of this function. So this is only needed if Pillow isn't at the
    latest version.

    :param image: The image to transpose.
    :type: ndarray
    :return: An image.
    :rtype: ndarray
    """
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    """Load an image file (.jpg, .png, etc) into a numpy array.

    Defaults to returning the image data as a 3-channel array of 8-bit data. That is
    controlled by the mode parameter.

    Supported modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)

    :param file: image file name or file object to load
    :type: str
    :param mode: format to convert the image to - 'RGB' (8-bit RGB, 3 channels), 'L' (black and white)
    :type: str
    :return: image contents as numpy array
    :rtype: ndarray
    """
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)


def imread(img_path):
    """Read image from image path.

    :param img_path
    :type: str
    :return: image array
    :rtype: nd.array
    """
    img_path = os.path.normpath(os.path.abspath(os.path.expanduser(img_path)))
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return img
        else:
            raise IOError(img_path)
    else:
        raise FileNotFoundError(img_path)


def get_img_whc(img):
    """Get image whc by src image.

    :param img: image to transform.
    :type: ndarray
    :return: image info
    :rtype: dict
    """
    img_shape = img.shape
    if len(img_shape) == 2:
        h, w = img_shape
        c = 1
    elif len(img_shape) == 3:
        h, w, c = img_shape
    else:
        raise NotImplementedError()
    return dict(width=w, height=h, channel=c)


def bgr2rgb(img):
    """Convert image from bgr type to rgb type.

    :param img: the image to be convert
    :type img: nd.array
    :return: the converted image
    :rtype: nd.array
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_lines(file_path):
    """Read multi-lines file to list.

    :param file_path: as name is the path of target file
    :type file_path: str
    :return: the content of file
    :rtype: list
    """
    with open(file_path) as f:
        target_lines = list(map(str.strip, f))
    return target_lines


def load_json(file_path):
    """Load annot json.

    :param file_path:file path
    :type: str
    :return:json content
    :rtype: dict
    """
    with open(file_path) as f:
        target_dict = json.load(f)
    return target_dict


def imagenet_normalize(*, img):
    """Normalize image.

    :param img: img that need to normalize
    :type img: RGB mode ndarray
    :return: normalized image
    :rtype: numpy.ndarray
    """
    pixel_value_range = np.array([255, 255, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / pixel_value_range
    img = img - mean
    img = img / std
    return img


def create_train_subset(data_path):
    """Create train dataset.

    :param data_path: path of data
    :type data_path: str
    """
    images_list_path = os.path.join(data_path, 'list', 'train.txt')
    images_list = load_lines(images_list_path)
    path_pairs = []
    for image_path_spec in images_list:
        path_pair_spec = dict(
            image_path=os.path.normpath(f'{data_path}/{image_path_spec}'),
            annot_path=os.path.normpath(f'{data_path}/{image_path_spec}'.replace('.jpg', '.lines.txt'))
        )
        path_pairs.append(path_pair_spec)
    return path_pairs


def create_valid_subset(data_path):
    """Create valid dataset.

    :param data_path: path of data
    :type data_path: str
    """
    images_list_path = os.path.join(data_path, 'list', 'val.txt')
    images_list = load_lines(images_list_path)
    path_pairs = []
    for image_path_spec in images_list:
        path_pair_spec = dict(
            image_path=os.path.normpath(f'{data_path}/{image_path_spec}'),
            annot_path=os.path.normpath(f'{data_path}/{image_path_spec}'.replace('.jpg', '.lines.txt'))
        )
        path_pairs.append(path_pair_spec)
    return path_pairs


def create_test_subset(data_path):
    """Create test dataset.

    :param data_path: path of data
    :type data_path: str
    """
    images_list_path = os.path.join(data_path, 'list', 'test.txt')
    images_list = load_lines(images_list_path)
    path_pairs = []
    for image_path_spec in images_list:
        path_pair_spec = dict(
            image_path=os.path.normpath(f'{data_path}/{image_path_spec}'),
            annot_path=os.path.normpath(f'{data_path}/{image_path_spec}'.replace('.jpg', '.lines.txt'))
        )
        path_pairs.append(path_pair_spec)
    return path_pairs
