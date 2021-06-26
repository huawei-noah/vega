# -*- coding:utf-8 -*-
from .pipe_step import PipeStep
from .pipeline import Pipeline
from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.core.pipeline", {
    "search_pipe_step": ["SearchPipeStep"],
    "train_pipe_step": ["TrainPipeStep"],
    "benchmark_pipe_step": ["BenchmarkPipeStep"],
    "multi_task_pipe_step": ["MultiTaskPipeStep"],
})
