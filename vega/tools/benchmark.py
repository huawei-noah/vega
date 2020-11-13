# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""
import vega
from vega.core.pipeline.benchmark_pipe_step import BenchmarkPipeStep
from vega.tools.init_env import _init_env
from vega.tools.args import _parse_args, _set_config


def _benchmark():
    args = _parse_args(["model"], "Benchmark.")
    vega.set_backend(args.general.backend)
    _set_config(args, "benchmark", "BenchmarkPipeStep")
    _init_env()
    BenchmarkPipeStep().do()


if __name__ == "__main__":
    _benchmark()
