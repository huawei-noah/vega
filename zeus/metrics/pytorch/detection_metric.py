# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of detection task by using coco tools."""
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from zeus.common import ClassFactory, ClassType
from zeus.metrics.pytorch.metrics import MetricBase


@ClassFactory.register(ClassType.METRIC, alias='coco')
class CocoMetric(MetricBase):
    """Save and summary metric from mdc dataset using coco tools."""

    __metric_name__ = "coco"

    def __init__(self, anno_path=None, category=None):
        self.anno_path = anno_path
        self.category = category
        self.result_record = []

    def __call__(self, output, targets, *args, **kwargs):
        """Append input into result record cache.

        :param output: output data
        :param target: target data
        :return:
        """
        if isinstance(output, dict):
            return None
        coco_results = []
        for id, prediction in enumerate(output):
            boxes = xyxy2xywh(prediction['boxes'])
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            img_id = targets[id]['image_id'].tolist()[0]
            for idx, box in enumerate(boxes):
                data = {}
                data['image_id'] = img_id
                data['bbox'] = box
                data['score'] = scores[idx]
                data['category_id'] = labels[idx]
                coco_results.append(data)
        self.result_record.extend(coco_results)
        return None

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.result_record = []

    def summary(self):
        """Summary all record from result cache, and get performance."""
        if not self.result_record:
            return {"mAP": -1, "AP_small": -1, "AP_medium": -1, "AP_large": -1}
        det_json_file = os.path.join(os.path.dirname(self.anno_path), 'det_json_file.json')
        with open(det_json_file, 'w') as f:
            json.dump(self.result_record, f)
        eval_result = self.print_scores(det_json_file, self.anno_path)
        eval_result = eval_result['AP(bbox)']
        eval_result = list(eval_result)
        return {
            "mAP": eval_result[1] * 100,
            "AP_small": eval_result[3] * 100,
            "AP_medium": eval_result[4] * 100,
            "AP_large": eval_result[5] * 100
        }

    def print_scores(self, det_json_file, json_file):
        """Print scores.

        :param det_json_file: dest json file
        :param json_file: gt json file
        :return:
        """
        ret = {}
        coco = COCO(json_file)
        cocoDt = coco.loadRes(det_json_file)
        cocoEval = COCOeval(coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ret['AP(bbox)'] = cocoEval.stats
        return ret


def xyxy2xywh(boxes):
    """Transform the bbox coordinate to [x,y ,w,h].

    :param bbox: the predict bounding box coordinate
    :type bbox: list
    :return: [x,y ,w,h]
    :rtype: list
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    import torch
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()
