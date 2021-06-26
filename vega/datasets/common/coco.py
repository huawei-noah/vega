# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for coco dataset."""
import logging
import os
import json
import numpy as np
import torch
from PIL import Image
from vega.common import ClassFactory, ClassType
from vega.datasets.conf.coco import CocoConfig
from vega.datasets.common.dataset import Dataset
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from vega.common.task_ops import TaskOps


@ClassFactory.register(ClassType.DATASET)
class CocoDataset(Dataset):
    """This is the class of coco dataset, which is a subclass of Dataset.

    :param train: `train`, `val` or test
    :type train: str
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = CocoConfig()

    def __init__(self, **kwargs):
        """Construct the CocoDataset class."""
        super(CocoDataset, self).__init__(**kwargs)
        self.dataset_init()

    def dataset_init(self):
        """Construct method."""
        self.num_classes = self.args.num_classes
        self.data_root = self.args.data_root
        img_prefix = str(self.args.img_prefix)
        ann_prefix = self.args.ann_prefix
        self.ann_file = os.path.join(self.data_root, 'annotations',
                                     '{}_{}{}.json'.format(ann_prefix, self.mode, img_prefix))
        self.img_dir = os.path.join(self.data_root, '{}{}'.format(self.mode, img_prefix))
        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # filter invalid data
        self.ids = list(filter(self._filter_invalid_ids, self.ids))
        self.collate_fn = collate_fn

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert('RGB')
        target = dict(image_id=img_id, annotations=target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.ids)

    def _filter_invalid_ids(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=0)
        anno = self.coco.loadAnns(ann_ids)
        return self._has_valid_annotation(anno)

    def _has_valid_annotation(self, anno):
        min_keypoints_per_image = 10

        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        # if it's empty, there is no annotation
        def _count_visible_keypoints(anno):
            return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

        if len(anno) == 0:
            return False
        if _has_only_empty_bbox(anno):
            return False
        if "keypoints" not in anno[0]:
            return True
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False


def collate_fn(batch):
    """Collate fn for data loader."""
    return tuple(zip(*batch))


@ClassFactory.register(ClassType.DATASET)
class DetectionDataset(Dataset):
    """Detection common dataset."""

    config = CocoConfig()

    def __init__(self, **kwargs):
        """Construct the Detection Dataset class."""
        super(DetectionDataset, self).__init__(**kwargs)
        self.imgs = list(sorted(os.listdir(os.path.join(self.args.data_root, self.args.img_prefix))))
        self.masks = list(sorted(os.listdir(os.path.join(self.args.data_root, self.args.ann_prefix))))
        portion = self.args.test_size
        self.imgs = self.imgs[:-portion] if self.mode == 'train' else self.imgs[-portion:]
        self.masks = self.masks[:-portion] if self.mode == 'train' else self.masks[-portion:]
        self.collate_fn = collate_fn
        convert_to_coco_api(self)

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index."""
        # load images and masks
        img_path = os.path.join(self.args.data_root, self.args.img_prefix, self.imgs[idx])
        mask_path = os.path.join(self.args.data_root, self.args.ann_prefix, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.imgs)


def convert_to_coco_api(ds):
    """Convert to coco dataset."""
    coco_ds = COCO()
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    instances_val = os.path.join(TaskOps().local_output_path, 'instances.json')
    json.dump(coco_ds.dataset, open(instances_val, 'w'))
    logging.info("dump detection instances json file: {}".format(instances_val))
    return coco_ds
