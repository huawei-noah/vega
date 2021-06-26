# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""CAM."""
import vega
import numpy as np
import cv2
import torch
from vega.common import argment_parser


def _predict_on_weights(feature_maps, weights):
    gap = np.average(feature_maps, axis=(0, 1))
    logit = np.dot(gap, np.squeeze(weights))
    return 1 / (1 + np.e ** (-logit))


def _get_cam(image, feature_maps, weights, display=False):
    predict = _predict_on_weights(feature_maps, weights)
    cam = (predict - 0.5) * np.matmul(feature_maps, weights)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    (width, height, channel) = image.shape
    cam = cv2.resize(np.array(cam), (width, height))
    cam = 255 * cam
    cam = cam.astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap[np.where(cam <= 100)] = 0
    image = 255 * image
    image = image.astype(np.uint8)
    out = cv2.addWeighted(src1=image, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)
    return out


def _load_image(image_file):
    img = cv2.imread(image_file)
    img = img / 255
    img = img.astype(np.float32)
    width, height, channel = img.shape
    return img.reshape(1, channel, height, width)


def _to_tensor(data):
    data = torch.tensor(data)
    return data.cuda()


def _get_model(args):
    from vega.model_zoo import ModelZoo
    model = ModelZoo.get_model(args.model_desc_file, args.model_weights_file)
    model = model.cuda()
    model.eval()
    return model


def _infer_pytorch(model, data):
    with torch.no_grad():
        logits = model(data)
        logits = logits.tolist()[0]
        return logits


def _hook(model, input, output):
    setattr(model, "feature_maps", input[0][0].cpu())


def _cam(args):
    img = _load_image(args.input_image_file)
    data = _to_tensor(img)
    model = _get_model(args)
    handle = next(model.head.children()).register_forward_hook(_hook)
    result = _infer_pytorch(model, data)
    handle.remove()
    cat = result.index(max(result))
    img = data[0].cpu().detach().numpy()
    channel, height, width = img.shape
    img = img.reshape(width, height, channel)
    feature_maps = next(model.head.children()).feature_maps
    channel, height, width = feature_maps.shape
    feature_maps = feature_maps.reshape(width, height, channel)
    weights = model.head.linear.weight[cat].cpu().detach().numpy()
    cam = _get_cam(img, feature_maps, weights)
    cv2.imwrite(args.output_image_file, cam)


def _parse_args():
    parser = argment_parser("Generate CAM(Class Activation Map) file.")
    parser.add_argument("-i", "--input_image_file", required=True, type=str, help="Input image file.")
    parser.add_argument("-o", "--output_image_file", required=True, type=str, help="Output image file.")
    parser.add_argument("-d", "--model_desc_file", required=True, type=str, help="Model description file.")
    parser.add_argument("-w", "--model_weights_file", required=True, type=str, help="Model weights file(.pth).")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    vega.set_backend("pytorch")
    args = _parse_args()
    print("model description file: {}".format(args.model_desc_file))
    print("model weights file: {}".format(args.model_weights_file))
    print("input image: {}".format(args.input_image_file))
    print("output image: {}".format(args.output_image_file))
    try:
        _cam(args)
        print("OK.")
    except Exception as e:
        raise e
