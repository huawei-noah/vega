# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Evaluate for coco."""

from mmdet.core.evaluation.coco_utils import *


def coco_eval(result_files, result_types, coco, dump_file=None):
    """Evaluate for coco metrics.

    :param result_files: prediction
    :type result_files: list
    :param result_types: evaluation types (bbox segm etc)
    :type result_types: str or list
    :param coco: coco data class
    :type coco: coco class
    :param dump_file: the file to save evaluation results
    :type dump_file: str
    """
    for res_type in result_types:
        assert res_type in ['bbox', 'segm']

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    results = dict()
    for res_type in result_types:
        if isinstance(result_files, str):
            result_file = result_files
        elif isinstance(result_files, dict):
            result_file = result_files[res_type]
        else:
            assert TypeError('result_files must be a str or dict')
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        cocoEval = COCOeval(coco, coco_dets, res_type)
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results.update({res_type: cocoEval.stats})

    if dump_file is not None:
        mmcv.dump(results, dump_file)
