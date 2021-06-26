from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.algorithms.auto_loss", {
    "adaptive_muti_loss": ["Autoloss"],
})
