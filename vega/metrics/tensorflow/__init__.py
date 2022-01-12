from vega.common.class_factory import ClassFactory
from .metrics import Metrics


ClassFactory.lazy_register("vega.metrics.tensorflow", {
    "segmentation_metric": ["trainer.metric:IoUMetric"],
    "classifier_metric": ["trainer.metric:accuracy"],
    "sr_metric": ["trainer.metric:PSNR", "trainer.metric:SSIM"],
    "forecast": ["trainer.metric:MSE", "trainer.metric:RMSE"],
    "r2score": ["trainer.metric:r2score", "trainer.metric:R2Score"],
})
