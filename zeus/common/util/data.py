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
"""Utils for saving data."""
from __future__ import absolute_import, division, print_function

import pickle

import h5py


def init_file(name):
    """
    Initilize file.

    :param name:
    :return:
    """
    try:
        model_file = h5py.File(name, 'r+')
        print("File is exsited")
    except BaseException:
        datatype = h5py.special_dtype(vlen=str)
        model_file = h5py.File(name, 'w')
        model_file.create_dataset("data_0", (5000, ), dtype=datatype)
        length = [0, 0]
        model_file.create_dataset("len", data=length)
        # print("create file")
    return model_file


def save_data(model_file, data):
    """
    Save data.

    :param model_file:
    :param data:
    :return:
    """
    index = model_file['len'][0]
    dataset_index = model_file['len'][1]

    if index >= (dataset_index + 1) * 5000:
        dataset_index += 1
        datatype = h5py.special_dtype(vlen=str)
        model_file.create_dataset("data_" + str(dataset_index), (5000, ), dtype=datatype)
        model_file['len'][1] = dataset_index

    model_file['data_' + str(dataset_index)][index % 5000] = pickle.dumps(data)
    index += 1
    model_file['len'][0] = index


def reset_data(model_file, datalen, dataset_index):
    """
    Reset data.

    :param model_file:
    :param datalen:
    :param dataset_index:
    :return:
    """
    model_file['len'][0] = datalen
    model_file['len'][1] = dataset_index


def get_datalen(model_file):
    """
    Get data length.

    :param model_file:
    :return:
    """
    return model_file['len'][0]


def get_data(model_file, index):
    """
    Get data.

    :param model_file:
    :param index:
    :return:
    """
    dataset_index = int(index / 5000)
    data = model_file['data_' + str(dataset_index)][index % 5000]
    return data.tostring()


def close_file(model_file):
    """
    Close file.

    :param model_file:
    :return:
    """
    model_file.close()
