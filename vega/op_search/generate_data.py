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

"""This script is used to generate data."""
import numpy as np
from utils import cal_mish, cal_gelu, cal_softplus, cal_tanh, cal_sqrt
from vega.common import FileOps

input_data = np.random.random([1, 3, 224, 224]).astype(np.float32) * 20 - 10
out_mish = cal_mish(input_data)
out_gelu = cal_gelu(input_data)
out_softplus = cal_softplus(input_data)
out_tanh = cal_tanh(input_data)
sqrt_input_data = np.random.random([1, 3, 224, 224]).astype(np.float32) * 20
out_sqrt = cal_sqrt(sqrt_input_data)
if __name__ == "__main__":
    FileOps.dump_pickle(input_data, "./input.pkl")
    FileOps.dump_pickle(out_mish, "./out_mish.pkl")
    FileOps.dump(out_gelu, "./out_gelu.pkl")
    FileOps.dump(out_gelu, "./out_softplus.pkl")
    FileOps.dump(out_tanh, "./out_tanh.pkl")
    FileOps.dump(out_sqrt, "./out_sqrt.pkl")
