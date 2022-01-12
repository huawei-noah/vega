# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""This is a class for coco dataset."""

import os
from PIL import Image
from vega.common import ClassFactory, ClassType
from vega.datasets.conf.coco import CocoConfig
from vega.datasets.common.dataset import Dataset
from pycocotools.coco import COCO


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
