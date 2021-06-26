from .metrics import Metrics
from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.metrics.mindspore", {
    "segmentation_metric": ["trainer.metric:IoUMetric"],
    "classifier_metric": ["trainer.metric:accuracy"],
    "sr_metric": ["trainer.metric:PSNR", "trainer.metric:SSIM"],
})
