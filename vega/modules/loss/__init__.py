from .loss import Loss
from vega.common.class_factory import ClassFactory

ClassFactory.lazy_register("vega.modules.loss", {
    "multiloss": ["trainer.loss:MultiLoss", "trainer.loss:SingleLoss"],
    "focal_loss": ["trainer.loss:FocalLoss"],
    "f1_loss": ["trainer.loss:F1Loss"],
    "forecast_loss": ["trainer.loss:ForecastLoss"],
    "mean_loss": ["trainer.loss:MeanLoss"],
    "ProbOhemCrossEntropy2d": ["trainer.loss:ProbOhemCrossEntropy2d"],
    "gan_loss": ["trainer.loss:GANLoss"],
})
