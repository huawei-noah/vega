# -*- coding:utf-8 -*-
from vega.common.class_factory import ClassFactory
from .pipe_step import PipeStep
from .pipeline import Pipeline


ClassFactory.lazy_register("vega.core.pipeline", {
    "search_pipe_step": ["SearchPipeStep"],
    "train_pipe_step": ["TrainPipeStep"],
    "benchmark_pipe_step": ["BenchmarkPipeStep"],
    "multi_task_pipe_step": ["MultiTaskPipeStep"],
    "horovod_train_step": ["HorovodTrainStep"],
    "hccl_train_step": ["HcclTrainStep"],
})
