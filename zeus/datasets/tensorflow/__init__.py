from zeus.common.class_factory import ClassFactory


ClassFactory.lazy_register("zeus.datasets.pytorch", {
    "coco_transforms": ["CocoCategoriesTransform", "PolysToMaskTransform"],
})
