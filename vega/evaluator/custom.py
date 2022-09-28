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

"""Custom Evaluator."""

import os
import numpy as np
from vega import ClassFactory, ClassType


@ClassFactory.register(ClassType.DEVICE_EVALUATOR)
class CustomEvaluator():
    """Define custom evaluator."""

    def __init__(self, device_evaluator):
        pass

    def get_data(self):
        """Get the evaluate data."""
        return np.random.random([1, 12, 320, 320]).astype(np.float32)

    def export_model(self, init_model):
        """Export the model to onnx/air and etc."""
        from mindspore.train.serialization import export
        from mindspore import Tensor
        from mindspore.common.api import _cell_graph_executor
        if hasattr(_cell_graph_executor, "set_jit_config"):
            _cell_graph_executor.set_jit_config(jit_config={"jit_level": "o0"})
        if hasattr(init_model, "set_jit_config"):
            from mindspore.common.jit_conig import JitConfig
            jit_conig = JitConfig(jit_level="O0")
            init_model.set_jit_config(jit_conig)
        fake_input = np.random.random([1, 12, 320, 320]).astype(np.float32)
        save_name = os.path.join("./", "ms2air.air")
        export(init_model, Tensor(fake_input), Tensor(640), file_name=save_name, file_format='AIR')
        model = save_name
        return model
