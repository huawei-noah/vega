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

"""This is a class for Exp1."""
from __future__ import absolute_import
from te import tvm
from topi import generic
import te.lang.cce
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType


def exp1_compute(input_x):
    """Compute function of the CusSquare implementation."""
    dtype = input_x.dtype
    x_mul = te.lang.cce.vmuls(input_x, tvm.const(0.25, dtype))
    add1 = te.lang.cce.vadds(x_mul, tvm.const(1, dtype))
    power2 = te.lang.cce.vmul(add1, add1)
    power4 = te.lang.cce.vmul(power2, power2)

    return power4


# Define the kernel info of CusSquare.
exp1_op_info = TBERegOp("Exp1") \
    .fusion_type("ELEMWISE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("exp1.so") \
    .compute_cost(10) \
    .kernel_name("Exp1Impl") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(exp1_op_info)
def Exp1Impl(input_x, output_y, kernel_name="Exp1Impl"):
    """Entry function of the CusSquare implementation."""
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res = exp1_compute(data)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
