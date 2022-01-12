from vega.common.class_factory import ClassFactory
from .loss import Loss

ClassFactory.lazy_register("vega.modules.loss", {
    "multiloss": ["trainer.loss:MultiLoss", "trainer.loss:SingleLoss"],
    "ProbOhemCrossEntropy2d": ["trainer.loss:ProbOhemCrossEntropy2d"],
    "ms_custom_loss": ["trainer.loss:CustomSoftmaxCrossEntropyWithLogits"],
})
