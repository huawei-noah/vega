from .metrics import Metrics
from zeus.common.class_factory import ClassFactory


ClassFactory.lazy_register("zeus.metrics.pytorch", {
    "lane_metric": ["trainer.metric:LaneMetric"],
    "regression": ["trainer.metric:MSE", "trainer.metric:mse"],
    "detection_metric": ["trainer.metric:CocoMetric", "trainer.metric:coco"],
    "gan_metric": ["trainer.metric:GANMetric"],
    "classifier_metric": ["trainer.metric:accuracy", "trainer.metric:Accuracy", "trainer.metric:SklearnMetrics"],
    "auc_metrics": ["trainer.metric:AUC", "trainer.metric:auc"],
    "segmentation_metric": ["trainer.metric:IoUMetric"],
    "sr_metric": ["trainer.metric:PSNR", "trainer.metric:SSIM"],
})
