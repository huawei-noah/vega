# -*- coding: utf-8 -*-
"""The tools to compare the output file with the benchmark."""
import struct
import numpy as np
import logging


def binary2list(binary_file, size, dtype):
    """Convert the binary file to a list.

    :param binary_file: the binary file name
    :type binary_file: str
    :param size: the size of the output
    :type size: int
    :param dtype: the dtype of the output
    :type dtype: str
    """
    FMT_MAP = {"f": 4, "q": 8}
    if dtype == "float32":
        fmt = "f"
    elif dtype == "int64":
        fmt = "q"
    else:
        raise ValueError("The dtype should be float32 or int64.")
    list_data = np.zeros(size)
    with open(binary_file, "rb") as f:
        for index in range(size):
            list_data[index] = struct.unpack(fmt, f.read(FMT_MAP[fmt]))[0]

    return list_data


def data_compare(real_out, expect_out, atol=0.001, rtol=0.001):
    """Compare the output between the real and the expect.

    :param real_out: the real output
    :type real_out: list
    :param expect_out: the expect putput, or the benchmark
    :type expect_out: list
    :param atol: the absolute error, defaults to 0.001
    :type atol: float, optional
    :param rtol: the relative error, defaults to 0.001
    :type rtol: float, optional
    :return: return the error count and the error ratio
    :rtype: [type]
    """
    error_count = 0
    if len(real_out) != len(expect_out):
        raise ValueError("The size of real_out and expect_out must be equal.")
    for n in range(len(real_out)):
        if abs(real_out[n] - expect_out[n]) > atol or abs(real_out[n] - expect_out[n]) / abs(expect_out[n]) > rtol:
            logging.warning("pos: {}, real_out: {}, expect_out: {}, diff: {} ".format(
                [n], real_out[n], expect_out[n], real_out[n] - expect_out[n]))
            error_count += 1
    return error_count, error_count / len(real_out)


if __name__ == "__main__":
    real_out_path = "./_output_0.bin"
    expect_out_path = "./expect_out_1.data"
    real_out = binary2list(real_out_path, 1001, "float32")
    expect_out = binary2list(expect_out_path, 1001, "float32")
    res = data_compare(real_out, expect_out)
    logging.warning("error_count:{}, error ratio: {}".format(res[0], res[1]))
