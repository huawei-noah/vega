# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ECP DATA API."""

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.registry import DATASETS
from pycocotools.coco import COCO


@DATASETS.register_module
class ECPDataset(CocoDataset):
    """ECPDataset."""

    CLASSES = ("pedestrian", "rider")

    def load_annotations(self, ann_file):
        """Load and initialize annotations."""
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        filter_imgs = []

        # filter some proposals
        ids_with_ann = set()
        for _ in self.coco.anns.values():
            # filter objects with others classes and area < 20
            if _['ignore'] is None or _['area'] < 20:
                continue
            # elif _['ignore'] >= 0: # only consider sample with 'ignore' == 0
            ids_with_ann.add(_['image_id'])

        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                filter_imgs.append(self.img_ids[i])
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        print("filter {} images".format(len(filter_imgs)))
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation."""
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.cat_ids)
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)
