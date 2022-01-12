# -*- coding:utf-8 -*-

# This file is adapted from the torchvision library at
# https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py

# 2020.11.12-Changed for vega
#       Huawei Technologies Co., Ltd. <chenchen6@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.

"""This is a class for Coco Transforms."""

import copy
import torch
from pycocotools import mask as coco_mask
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class CocoCategoriesTransform(object):
    """Filter coco dataset by categories."""

    def __init__(self, categories, remap=True):
        """Construct the CocoCategoriesTransform class."""
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        """Filter annotations by category_id."""
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


@ClassFactory.register(ClassType.TRANSFORM)
class PolysToMaskTransform(object):
    """Polys to mask."""

    def __call__(self, image, target):
        """Convert poly to mask."""
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert coco poly to mask."""
    masks = []
    for polygons in segmentations:
        if not polygons:
            continue
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


@ClassFactory.register(ClassType.TRANSFORM)
class PrepareVOCInstance(object):
    """Convert dataset to Voc instance."""

    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, classes=None):
        self.classes = classes or self.CLASSES
        if isinstance(self.classes, list):
            self.classes = tuple(self.classes)

    def __call__(self, image, target):
        """Convert to voc."""
        anno = target['annotation']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.classes.index(obj['name']))
            iscrowd.append(int(obj['difficult']))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)
        image_id = anno['filename'][5:-4]
        image_id = image_id
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["file_name"] = anno['filename']
        return image, target
