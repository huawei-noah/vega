from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.datasets.pytorch", {
    "coco_transforms": ["CocoCategoriesTransform", "PolysToMaskTransform"],
})
