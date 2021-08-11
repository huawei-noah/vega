from .callback import Callback
from .callback_list import CallbackList
from vega.common.class_factory import ClassFactory


__all__ = ["Callback", "CallbackList"]


ClassFactory.lazy_register("vega.trainer.callbacks", {
    "metrics_evaluator": ["trainer.callback:MetricsEvaluator"],
    "progress_logger": ["trainer.callback:ProgressLogger"],
    "performance_saver": ["trainer.callback:PerformanceSaver"],
    "lr_scheduler": ["trainer.callback:LearningRateScheduler"],
    "model_builder": ["trainer.callback:ModelBuilder"],
    "model_statistics": ["trainer.callback:ModelStatistics"],
    "model_checkpoint": ["trainer.callback:ModelCheckpoint"],
    "report_callback": ["trainer.callback:ReportCallback"],
    "runtime_callback": ["trainer.callback:RuntimeCallback"],
    "detection_progress_logger": ["trainer.callback:DetectionProgressLogger"],
    "detection_metrics_evaluator": ["trainer.callback:DetectionMetricsEvaluator"],
    "visual_callback": ["trainer.callback:VisualCallBack"],
    "model_tuner": ["trainer.callback:ModelTuner"],
    "timm_trainer_callback": ["trainer.callback:TimmTrainerCallback"],
    "data_parallel": ["trainer.callback:DataParallel"],
})
