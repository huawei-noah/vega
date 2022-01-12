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

"""This is a class for Test."""
import time
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from exp1 import Exp1
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class My_exp(nn.Cell):
    """Mish activation method."""

    def __init__(self):
        super(My_exp, self).__init__()
        self.exp1 = Exp1()

    def construct(self, x):
        """Forward Mish."""
        return self.exp1(x)


class Init_exp(nn.Cell):
    """Mish activation method."""

    def __init__(self):
        super(Init_exp, self).__init__()
        self.exp = P.Exp()

    def construct(self, x):
        """Forward Mish."""
        return self.exp(x)


class My_Mish(nn.Cell):
    """Mish activation method."""

    def __init__(self):
        super(My_Mish, self).__init__()
        self.exp1 = Exp1()
        self.pow = P.Pow()

    def construct(self, x):
        """Forward Mish."""
        return x * (1 - 2 / (self.pow(1 + self.exp1(x), 2) + 1))


class Init_Mish(nn.Cell):
    """Mish net definition."""

    def __init__(self):
        super(Init_Mish, self).__init__()
        self.mish = P.Mish()

    def construct(self, x):
        """Forward Mish."""
        out = self.mish(x)
        return out


class Splice_Mish(nn.Cell):
    """Mish activation method."""

    def __init__(self):
        super(Splice_Mish, self).__init__()
        self.mul = P.Mul()
        self.tanh = P.Tanh()
        self.softplus = P.Softplus()

    def construct(self, input_x):
        """Forward Mish."""
        res1 = self.softplus(input_x)
        tanh = self.tanh(res1)
        output = self.mul(input_x, tanh)
        return output


repeat_times = 10000

data_size = 32 * 64 * 112 * 112
test_data = np.linspace(num=data_size, start=-10, stop=10).astype(np.float32)
test_data = test_data.reshape(32, 64, 112, 112)
print(test_data.shape)

print("the max/min of input:", np.max(test_data), np.min(test_data))
test_tensor = Tensor(test_data)

net = Init_exp()
start_time = time.time()
for i in range(repeat_times):
    if i == 1:
        print("the compile time is:", time.time() - start_time)
        out2 = net(test_tensor)
        start_time = time.time()
    else:
        out2 = net(test_tensor)

end_time = time.time()

print("cost time of exp(impl by default):", end_time - start_time)

net = My_exp()
start_time = time.time()
for i in range(repeat_times):
    if i == 1:
        print("the compile time is:", time.time() - start_time)
        out1 = net(test_tensor)
        start_time = time.time()
    else:
        out1 = net(test_tensor)

end_time = time.time()

print("cost time of exp(impl by me):", end_time - start_time)

net = Init_Mish()
start_time = time.time()
for i in range(repeat_times):
    if i == 1:
        print("the compile time is:", time.time() - start_time)
        out4 = net(test_tensor)
        start_time = time.time()
    else:
        out4 = net(test_tensor)

end_time = time.time()

print("cost time of mish(impl by default):", end_time - start_time)

net = Splice_Mish()
start_time = time.time()
for i in range(repeat_times):
    if i == 1:
        print("the compile time is:", time.time() - start_time)
        out3 = net(test_tensor)
        start_time = time.time()
    else:
        out3 = net(test_tensor)

end_time = time.time()

print("cost time of mish(impl by splice):", end_time - start_time)

net = My_Mish()
start_time = time.time()
for i in range(repeat_times):
    if i == 1:
        print("the compile time is:", time.time() - start_time)
        out3 = net(test_tensor)
        start_time = time.time()
    else:
        out3 = net(test_tensor)

end_time = time.time()

print("cost time of mish(impl by me):", end_time - start_time)
