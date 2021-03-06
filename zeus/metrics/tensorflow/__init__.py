from .metrics import Metrics
from zeus.common.class_factory import ClassFactory


ClassFactory.lazy_register("zeus.metrics.tensorflow", {
    "segmentation_metric": ["trainer.metric:IoUMetric"],
    "classifier_metric": ["trainer.metric:accuracy"],
    "sr_metric": ["trainer.metric:PSNR", "trainer.metric:SSIM"],
    "forecast": ["trainer.metric:MSE", "trainer.metric:RMSE"],
})
