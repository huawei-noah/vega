# -*- coding:utf-8 -*-

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

"""Metric of detection task by using coco tools."""
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from vega.common import ClassFactory, ClassType
from vega.metrics.mindspore.metrics import MetricBase
from vega.common.task_ops import TaskOps


@ClassFactory.register(ClassType.METRIC, alias='coco')
class CocoMetric(MetricBase):
    """Save and summary metric from mdc dataset using coco tools."""

    __metric_name__ = "coco"

    def __init__(self, anno_path=None, category=None):
        self.anno_path = anno_path or os.path.join(TaskOps().local_output_path, 'instances.json')
        self.category = category or []
        self.result_record = []

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return {'mAP': 'MAX', 'AP50': 'MAX', 'AP_small': 'MAX', 'AP_medium': 'MAX', 'AP_large': 'MAX'}

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
            scores = prediction["scores"].asnumpy().tolist()
            labels = prediction["labels"].asnumpy().tolist()
            img_id = targets[id]['image_id'].asnumpy().tolist()[0]
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
        det_json_file = os.path.join(TaskOps().local_output_path, 'det_json_file.json')
        with open(det_json_file, 'w') as f:
            json.dump(self.result_record, f)
        eval_result = self.print_scores(det_json_file, self.anno_path)
        ap_result = eval_result.pop('AP(bbox)')
        ap_result = list(ap_result)
        ap_result = {
            "mAP": ap_result[0] * 100,
            "AP50": ap_result[1] * 100,
            "AP_small": ap_result[3] * 100,
            "AP_medium": ap_result[4] * 100,
            "AP_large": ap_result[5] * 100
        }
        if eval_result:
            ap_result.update(eval_result)
        return ap_result

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
        for id, item in enumerate(self.category):
            cocoEval = COCOeval(coco, cocoDt, 'bbox')
            cocoEval.params.catIds = [id + 1]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if len(cocoEval.stats) > 0:
                ret[item] = cocoEval.stats[1] * 100
        return ret


def xyxy2xywh(boxes):
    """Transform the bbox coordinate to [x,y ,w,h].

    :param bbox: the predict bounding box coordinate
    :type bbox: list
    :return: [x,y ,w,h]
    :rtype: list
    """
    from mindspore import ops
    ms_unbind = ops.Unstack(axis=1)
    ms_stack = ops.Stack(axis=1)
    xmin, ymin, xmax, ymax = ms_unbind(boxes)
    return ms_stack((xmin, ymin, xmax - xmin, ymax - ymin)).asnumpy().tolist()
