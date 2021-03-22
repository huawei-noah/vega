from .loss import Loss
from zeus.common.class_factory import ClassFactory

ClassFactory.lazy_register("zeus.modules.loss", {
    "multiloss": ["trainer.loss:MultiLoss"],
    "focal_loss": ["trainer.loss:FocalLoss"],
    "f1_loss": ["trainer.loss:F1Loss"],
    "forecast_loss": ["trainer.loss:ForecastLoss"],
    "mean_loss": ["trainer.loss:MeanLoss"],
    "ProbOhemCrossEntropy2d": ["trainer.loss:ProbOhemCrossEntropy2d"],
    "gan_loss": ["trainer.loss:GANLoss"],
})
