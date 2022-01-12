# -*- coding=utf-8 -*-

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

"""This is the class for CurveLane dataset."""

import json
import numpy as np
from more_itertools import grouper
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.lines import LineString as ia_LineString
import imgaug as ia
import imgaug.augmenters as iaa
from vega.common import ClassFactory, ClassType
from vega.datasets.common.utils.auto_lane_pointlane_codec import PointLaneCodec
from vega.datasets.conf.auto_lane import AutoLaneConfig
from vega.common import FileOps
from .dataset import Dataset
from .utils.auto_lane_utils import get_img_whc, imread, create_train_subset, create_test_subset
from .utils.auto_lane_utils import load_lines, resize_by_wh, bgr2rgb, imagenet_normalize, load_json


def _culane_line_to_curvelane_dict(culane_lines):
    curvelane_lines = []
    for culane_line_spec in culane_lines:
        curvelane_lien_spec = [{'x': x, 'y': y} for x, y in grouper(map(float, culane_line_spec.split(' ')), 2)]
        curvelane_lines.append(curvelane_lien_spec)
    return dict(Lines=curvelane_lines)


def _lane_argue(*, image, lane_src):
    lines_tuple = [[(float(pt['x']), float(pt['y'])) for pt in line_spec] for line_spec in lane_src['Lines']]
    lss = [ia_LineString(line_tuple_spec) for line_tuple_spec in lines_tuple]

    lsoi = LineStringsOnImage(lss, shape=image.shape)
    color_shift = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.5, 1.5)),
        iaa.LinearContrast((1.5, 1.5), per_channel=False),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(0, iaa.Multiply((0.7, 1.3)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(1, iaa.Multiply((0.1, 2)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),
    ])
    posion_shift = iaa.SomeOf(4, [
        iaa.Fliplr(),
        iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [0, 0.15]), keep_size=True),
        iaa.TranslateX(px=(-16, 16)),
        iaa.ShearX(shear=(-15, 15)),
        iaa.Rotate(rotate=(-15, 15))
    ])
    aug = iaa.Sequential([
        iaa.Sometimes(p=0.6, then_list=color_shift),
        iaa.Sometimes(p=0.6, then_list=posion_shift)
    ], random_order=True)
    batch = ia.Batch(images=[image], line_strings=[lsoi])
    batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
    image_aug = batch_aug.images_aug[0]
    lsoi_aug = batch_aug.line_strings_aug[0]
    lane_aug = [[dict(x=kpt.x, y=kpt.y) for kpt in shapely_line.to_keypoints()] for shapely_line in lsoi_aug]
    return image_aug, dict(Lines=lane_aug)


def _read_curvelane_type_annot(annot_path):
    return load_json(annot_path)


def _read_culane_type_annot(annot_path):
    return _culane_line_to_curvelane_dict(load_lines(annot_path))


@ClassFactory.register(ClassType.DATASET)
class AutoLaneDataset(Dataset):
    """This is the class of CurveLane dataset, which is a subclass of Dataset.

    :param train: `train`, `val` or `test`
    :type train: str
    :param cfg: config of this datatset class
    :type cfg: yml file that in the entry
    """

    config = AutoLaneConfig()

    def __init__(self, **kwargs):
        """Construct the dataset."""
        super().__init__(**kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        dataset_pairs = dict(
            train=create_train_subset(self.args.data_path),
            test=create_test_subset(self.args.data_path),
            val=create_test_subset(self.args.data_path)
        )

        if self.mode not in dataset_pairs.keys():
            raise NotImplementedError(f'mode should be one of {dataset_pairs.keys()}')
        self.image_annot_path_pairs = dataset_pairs.get(self.mode)

        self.codec_obj = PointLaneCodec(input_width=512, input_height=288,
                                        anchor_stride=16, points_per_line=72,
                                        class_num=2)
        self.encode_lane = self.codec_obj.encode_lane
        read_funcs = dict(
            CULane=_read_culane_type_annot,
            CurveLane=_read_curvelane_type_annot,
        )
        if self.args.dataset_format not in read_funcs:
            raise NotImplementedError(f'dataset_format should be one of {read_funcs.keys()}')
        self.read_annot = read_funcs.get(self.args.dataset_format)
        self.with_aug = self.args.get('with_aug', False)

    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return len(self.image_annot_path_pairs)

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        if self.mode == 'train':
            return self.prepare_train_img(idx)
        elif self.mode == 'val':
            return self.prepare_test_img(idx)
        elif self.mode == 'test':
            return self.prepare_test_img(idx)
        else:
            raise NotImplementedError

    def prepare_train_img(self, idx):
        """Prepare an image for training.

        :param idx:index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]
        if self.with_aug:
            try:
                lane_object = self.read_annot(target_pair['annot_path'])
                image_arr = imread(target_pair['image_path'])
                whc = get_img_whc(image_arr)
                image_arr, lane_object = _lane_argue(image=image_arr, lane_src=lane_object)
                encode_type, encode_loc = self.encode_lane(lane_object=lane_object,
                                                           org_width=whc['width'],
                                                           org_height=whc['height'])
            except Exception:
                lane_object = self.read_annot(target_pair['annot_path'])
                image_arr = imread(target_pair['image_path'])
                whc = get_img_whc(image_arr)
                encode_type, encode_loc = self.encode_lane(lane_object=lane_object,
                                                           org_width=whc['width'],
                                                           org_height=whc['height'])
        else:
            lane_object = self.read_annot(target_pair['annot_path'])
            image_arr = imread(target_pair['image_path'])
            whc = get_img_whc(image_arr)
            encode_type, encode_loc = self.encode_lane(lane_object=lane_object,
                                                       org_width=whc['width'],
                                                       org_height=whc['height'])

        network_input_image = bgr2rgb(resize_by_wh(img=image_arr, width=512, height=288))
        item = dict(
            net_input_image=imagenet_normalize(img=network_input_image),
            net_input_image_mode='RGB',
            net_input_image_shape=dict(width=512, height=288, channel=3),
            src_image_shape=whc,
            src_image_path=target_pair['image_path'],
            annotation_path=target_pair['annot_path'],
            annotation_src_content=lane_object,
            regression_groundtruth=encode_loc,
            classfication_groundtruth=encode_type
        )
        result = dict(image=np.transpose(item['net_input_image'], (2, 0, 1)).astype('float32'),
                      gt_loc=item['regression_groundtruth'].astype('float32'),
                      gt_cls=item['classfication_groundtruth'].astype('float32'))
        return result

    def prepare_test_img(self, idx):
        """Prepare an image for testing.

        :param idx: index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]
        image_arr = imread(target_pair['image_path'])
        lane_object = self.read_annot(target_pair['annot_path'])
        whc = get_img_whc(image_arr)
        network_input_image = bgr2rgb(resize_by_wh(img=image_arr, width=512, height=288))
        item = dict(
            net_input_image=imagenet_normalize(img=network_input_image),
            net_input_image_mode='RGB',
            net_input_image_shape=dict(width=512, height=288, channel=3),
            src_image_shape=whc,
            src_image_path=target_pair['image_path'],
            annotation_path=target_pair['annot_path'],
            annotation_src_content=lane_object,
            regression_groundtruth=None,
            classfication_groundtruth=None
        )

        result = dict(image=np.transpose(item['net_input_image'], (2, 0, 1)).astype('float32'),
                      net_input_image_shape=json.dumps(item['net_input_image_shape']),
                      src_image_shape=json.dumps(item['src_image_shape']),
                      annot=json.dumps(item['annotation_src_content']),
                      src_image_path=item['src_image_path'],
                      annotation_path=item['annotation_path'])

        return result
