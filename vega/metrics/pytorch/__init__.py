from vega.common.class_factory import ClassFactory
from .metrics import Metrics

ClassFactory.lazy_register("vega.metrics.pytorch", {
    "lane_metric": ["trainer.metric:LaneMetric"],
    "regression": ["trainer.metric:MSE", "trainer.metric:mse"],
    "detection_metric": ["trainer.metric:CocoMetric", "trainer.metric:coco"],
    "classifier_metric": ["trainer.metric:accuracy", "trainer.metric:Accuracy", "trainer.metric:SklearnMetrics"],
    "auc_metrics": ["trainer.metric:AUC", "trainer.metric:auc"],
    "segmentation_metric": ["trainer.metric:IoUMetric"],
    "sr_metric": ["trainer.metric:PSNR", "trainer.metric:SSIM"],
    "r2score": ["trainer.metric:r2score", "trainer.metric:R2Score"],
    "nlp_metric": ["trainer.metric:accuracy_score", "trainer.metric:f1_score", "trainer.metric:NlpMetrics"],
})
