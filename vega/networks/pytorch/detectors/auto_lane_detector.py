# -*- coding: utf-8 -*-
"""Defined faster rcnn detector."""
from collections import ChainMap
import torch
from torch import nn
from torch.nn import functional as F
import ujson
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory
from vega.datasets.common.utils.auto_lane_pointlane_codec import PointLaneCodec
from vega.datasets.common.utils.auto_lane_codec_utils import nms_with_pos, order_lane_x_axis
from vega.datasets.common.utils.auto_lane_codec_utils import convert_lane_to_dict


def get_img_whc(img):
    """Get image whc by src image.

    :param img: image to transform.
    :type: ndarray
    :return: image info
    :rtype: dict
    """
    img_shape = img.shape
    if len(img_shape) == 2:
        h, w = img_shape
        c = 1
    elif len(img_shape) == 3:
        h, w, c = img_shape
    else:
        raise NotImplementedError()
    return dict(width=w, height=h, channel=c)


def multidict_split(bundle_dict):
    """Split multi dict to retail dict.

    :param bundle_dict: a buddle of dict
    :type bundle_dict: a dict of list
    :return: retails of dict
    :rtype: list
    """
    retails_list = [dict(zip(bundle_dict, i)) for i in zip(*bundle_dict.values())]
    return retails_list


def find_k_th_small_in_a_tensor(target_tensor, k_th):
    """Like name, this function will return the k the of the tensor."""
    val, idxes = torch.topk(target_tensor, k=k_th, largest=False)
    return val[-1]


def huber_fun(x):
    """Implement of hunber function."""
    absx = torch.abs(x)
    r = torch.where(absx < 1, x * x / 2, absx - 0.5)
    return r


@ClassFactory.register(ClassType.NETWORK)
class AutoLaneDetector(Module):
    """Faster RCNN."""

    def __init__(self, desc):
        """Init faster rcnn.

        :param desc: config dict
        """
        super(AutoLaneDetector, self).__init__()
        self.desc = desc
        self.num_class = int(desc["num_class"])
        self.pointlane_codec = PointLaneCodec(input_width=512,
                                              input_height=288,
                                              anchor_stride=16,
                                              points_per_line=72,
                                              class_num=2)

        def build_module(net_type_name):
            return ClassFactory.get_cls(ClassType.NETWORK, desc[net_type_name].type)(desc[net_type_name])

        self.backbone = build_module('backbone')
        self.neck = build_module('neck')
        self.head = build_module('head')
        self.LANE_POINTS_NUM_DOWN = 72
        self.LANE_POINTS_NUM_UP = 73
        self.LANE_POINT_NUM_GT = 145
        self.ALPHA = 10
        self.NEGATIVE_RATIO = 15
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_feat(self, img):
        """Public compute of input.

        :param img: input image
        :return: feature map after backbone and neck compute
        """
        x = self.backbone(img)
        x = self.neck(x[0:4])
        return x

    def forward(self, input, forward_switch='calc', **kwargs):
        """Call default forward function."""
        if forward_switch == 'train':
            return self.forward_train(input, **kwargs)
        elif forward_switch == 'valid':
            return self.forward_valid(input, **kwargs)
        elif forward_switch == 'calc':
            return self.forward_calc_params_and_flops(input, **kwargs)

    def forward_calc_params_and_flops(self, input, **kwargs):
        """Just for calc paramters."""
        feat = self.extract_feat(input)
        predict = self.head(feat)
        return predict

    def forward_train(self, input, **kwargs):
        """Forward compute between train process.

        :param input: input data
        :return: losses
        :rtype: torch.tensor
        """
        image = input
        loc_targets = kwargs['gt_loc']
        cls_targets = kwargs['gt_cls']

        feat = self.extract_feat(image)
        predict = self.head(feat)

        loc_preds = predict['predict_loc']
        cls_preds = predict['predict_cls']
        cls_targets = cls_targets[..., 1].view(-1)
        pmask = cls_targets > 0
        nmask = ~ pmask
        fpmask = pmask.float()
        fnmask = nmask.float()
        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        loc_preds = loc_preds.view(-1, loc_preds.shape[-1])
        loc_targets = loc_targets.view(-1, loc_targets.shape[-1])
        total_postive_num = torch.sum(fpmask)
        total_negative_num = torch.sum(fnmask)  # Number of negative entries to select
        negative_num = torch.clamp(total_postive_num * self.NEGATIVE_RATIO, max=total_negative_num, min=1).int()
        positive_num = torch.clamp(total_postive_num, min=1).int()
        # cls loss begin
        bg_fg_predict = F.log_softmax(cls_preds, dim=-1)
        fg_predict = bg_fg_predict[..., 1]
        bg_predict = bg_fg_predict[..., 0]
        max_hard_pred = find_k_th_small_in_a_tensor(bg_predict[nmask].detach(), negative_num)
        fnmask_ohem = (bg_predict <= max_hard_pred).float() * nmask.float()
        total_cross_pos = -torch.sum(self.ALPHA * fg_predict * fpmask)
        total_cross_neg = -torch.sum(self.ALPHA * bg_predict * fnmask_ohem)
        # class loss end
        # regression loss begin
        length_weighted_mask = torch.ones_like(loc_targets)
        length_weighted_mask[..., self.LANE_POINTS_NUM_DOWN] = 10
        valid_lines_mask = pmask.unsqueeze(-1).expand_as(loc_targets)
        valid_points_mask = (loc_targets != 0)
        unified_mask = length_weighted_mask.float() * valid_lines_mask.float() * valid_points_mask.float()
        smooth_huber = huber_fun(loc_preds - loc_targets) * unified_mask
        loc_smooth_l1_loss = torch.sum(smooth_huber, -1)
        point_num_per_gt_anchor = torch.sum(valid_points_mask.float(), -1).clamp(min=1)
        total_loc = torch.sum(loc_smooth_l1_loss / point_num_per_gt_anchor)
        # regression loss end
        total_cross_pos = total_cross_pos / positive_num
        total_cross_neg = total_cross_neg / positive_num
        total_loc = total_loc / positive_num

        return dict(
            loss_pos=total_cross_pos,
            loss_neg=total_cross_neg,
            loss_loc=total_loc
        )

    def forward_valid(self, input, **kwargs):
        """Forward compute between inference.

        :param input: input data must be image
        :return: groundtruth result and predict result
        :rtype: dict
        """
        image = input
        feat = self.extract_feat(image)
        predict = self.head(feat)
        predict_result = dict(
            image=image.permute((0, 2, 3, 1)).detach().contiguous().cpu().numpy(),
            regression=predict['predict_loc'].detach().cpu().numpy(),
            classfication=F.softmax(predict['predict_cls'], -1).detach().cpu().numpy(),
        )

        bundle_result = ChainMap(kwargs, predict_result)
        results = []
        for index, retail_dict_spec in enumerate(multidict_split(bundle_result)):
            lane_set = self.pointlane_codec.decode_lane(
                predict_type=retail_dict_spec['classfication'],
                predict_loc=retail_dict_spec['regression'], cls_thresh=0.6)
            lane_nms_set = nms_with_pos(lane_set, thresh=60)

            net_input_image_shape = ujson.loads(retail_dict_spec['net_input_image_shape'])
            src_image_shape = ujson.loads(retail_dict_spec['src_image_shape'])
            lane_order_set = order_lane_x_axis(lane_nms_set, net_input_image_shape['height'])
            scale_x = src_image_shape['width'] / net_input_image_shape['width']
            scale_y = src_image_shape['height'] / net_input_image_shape['height']

            predict_json = convert_lane_to_dict(lane_order_set, scale_x, scale_y)
            target_json = ujson.loads(retail_dict_spec['annot'])
            results.append(dict(pr_result={**predict_json, **dict(Shape=src_image_shape)},
                                gt_result={**target_json, **dict(Shape=src_image_shape)}))
        return results
