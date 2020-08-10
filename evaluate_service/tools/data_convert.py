# -*- coding: utf-8 -*-
"""The tools to convert the binary data to numpy array."""
import numpy as np
import struct

FMT_MAP = {"f": 4, "q": 8}


def binary2np(binary_file, shape, dtype):
    """Convert the binary to numpy array.

    :param binary_file: the binary file name
    :type binary_file: .bin or .data
    :param shape: the arrary shape of the arrary
    :type shape: list
    :param dtype: the dtype of the output data
    :type dtype: str
    """
    dim_size = len(shape)
    np_data = np.zeros(shape)
    fmt = fmt_map(dtype)
    with open(binary_file, "rb") as f:
        if dim_size == 1:
            for dim1 in range(shape[0]):
                np_data[dim1] = struct.unpack(fmt, f.read(FMT_MAP[fmt]))[0]
        elif dim_size == 2:
            for dim1 in range(shape[0]):
                for dim2 in range(shape[1]):
                    np_data[dim1, dim2] = struct.unpack(fmt, f.read(FMT_MAP[fmt]))[0]
        elif dim_size == 3:
            for dim1 in range(shape[0]):
                for dim2 in range(shape[1]):
                    for dim3 in range(shape[2]):
                        np_data[dim1, dim2, dim3] = struct.unpack(fmt, f.read(FMT_MAP[fmt]))[0]
        elif dim_size == 4:
            np_data = binary2np_extra(binary_file, shape, dtype)
        else:
            raise ValueError("The dim size should be less than 5.")
    return np_data


def binary2np_extra(binary_file, shape, dtype):
    """Convert the binary to numpy array.

    :param binary_file: the binary file name
    :type binary_file: .bin or .data
    :param shape: the arrary shape of the arrary
    :type shape: list
    :param dtype: the dtype of the output data
    :type dtype: str
    """
    np_data = np.zeros(shape)
    fmt = fmt_map(dtype)
    with open(binary_file, "rb") as f:
        for dim1 in range(shape[0]):
            for dim2 in range(shape[1]):
                for dim3 in range(shape[2]):
                    for dim4 in range(shape[3]):
                        np_data[dim1, dim2, dim3, dim4] = struct.unpack(fmt, f.read(FMT_MAP[fmt]))[0]
    return np_data


def fmt_map(dtype):
    """Map dataformat.

    :param dtype: the dtype of the data
    :type dtype: str
    :return: the mapping format
    :rtype: str
    """
    if dtype == "float32":
        fmt = "f"
    elif dtype == "int64":
        fmt = "q"
    else:
        raise ValueError("The dtype should be float32 or int64.")
    return fmt
