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

"""Human-in-the-loop Optimizer, used for debugging."""
from collections import OrderedDict
from modnas.registry.optim import register
from modnas.optim.base import OptimBase
from modnas.core.params import Numeric


@register
class HITLOptim(OptimBase):
    """Human-in-the-loop Optimizer class."""

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        return True

    def parse_input(self, param, inp):
        """Return parsed value from input string."""
        if isinstance(param, Numeric):
            return float(inp)
        try:
            return int(inp)
        except ValueError:
            return inp

    def check_value(self, param, value):
        """Return True if the value is valid for the given parameter."""
        if value is None:
            return False
        return param.is_valid(value)

    def _next(self):
        """Return the next set of parameters."""
        ret = OrderedDict()
        for name, param in self.space.named_params():
            prompt = '{}\nvalue: '.format(str(param))
            while True:
                inp = input(prompt)
                value = self.parse_input(param, inp)
                if self.check_value(param, value):
                    break
                print('invalid input')
            ret[name] = value
        return ret

    def step(self, estim):
        """Update Optimizer states using Estimator evaluation results."""
        inputs, results = estim.get_last_results()
        self.logger.info('update:\ninputs: {}\nresults: {}'.format(inputs, results))
