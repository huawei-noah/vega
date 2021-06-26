# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of object detection model."""
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
from vega.model_zoo.model_zoo import ModelZoo


class ObjectDetectionAPI(object):
    """ObjectDetection API."""

    def __init__(self, desc_file, pretrained_model_file, threshold=0.5):
        super().__init__()
        self.model = ModelZoo().get_model(desc_file, pretrained_model_file)
        self.threshold = threshold

    def predict(self, img_path, category_names):
        """Predict one img."""
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        self.model.eval()
        pred = self.model([img])
        pred_class = [category_names[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        self._show(pred_boxes, pred_class, img_path)
        return pred_boxes, pred_class

    def _show(self, boxes, pred_cls, img_path, rect_th=3, text_size=3, text_th=3):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, box in enumerate(boxes):
            cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], box[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
