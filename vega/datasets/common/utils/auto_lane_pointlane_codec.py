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

"""This script is used to process the auto lane dataset."""

import numpy as np
from scipy import interpolate
from .auto_lane_spline_interp import spline_interp
from .auto_lane_codec_utils import Point, Lane, get_lane_list, delete_repeat_y
from .auto_lane_codec_utils import delete_nearby_point, trans_to_lane_with_type
from .auto_lane_codec_utils import get_lane_loc_list, gettopk_idx


class PointLaneCodec(object):
    """This is the class of PointLaneCodec, which generate the groudtruth of every image pair.

    :param input_width: the width of input image
    :type input_width: float
    :param input_height: the height of input image
    :type input_height: float
    :param anchor_stride: the stride of every anchor
    :type anchor_stride: int
    :param points_per_line: the number of points in line
    :type points_per_line: int
    :param class_num: the class num of all lines
    :type class_num: int
    :param anchor_lane_num: how many lanes of every anchor
    :type anchor_lane_num: int
    """

    def __init__(self, input_width, input_height, anchor_stride,
                 points_per_line, class_num, anchor_lane_num=1):
        self.input_width = input_width
        self.input_height = input_height
        self.feature_width = int(input_width / anchor_stride)
        self.feature_height = int(input_height / anchor_stride)
        self.points_per_line = points_per_line
        self.pt_nums_single_lane = 2 * points_per_line + 1
        self.points_per_anchor = points_per_line / self.feature_height
        self.interval = float(input_height) / points_per_line
        self.feature_size = self.feature_width * self.feature_height
        self.class_num = class_num
        self.img_center_x = input_width / 2
        self.step_w = anchor_stride
        self.step_h = anchor_stride
        self.anchor_lane_num = anchor_lane_num

    def encode_lane(self, lane_object, org_width, org_height):
        """Encode lane to target type.

        :param lane_object: lane annot
        :type lane_object: mon dict (a dict of special format)
        :param org_width: image width
        :type org_width: int
        :param org_height: image height
        :type org_height: int
        :return: gt_type: [576, class_num]
                 gt_loc:  [576, 145]
        :rtype: nd.array
        """
        s_x = self.input_width * 1.0 / org_width
        s_y = self.input_height * 1.0 / org_height
        gt_lanes_list = get_lane_list(lane_object, s_x, s_y)
        if len(gt_lanes_list) < 1:
            gt_lane_offset = np.zeros(shape=(self.feature_size, self.points_per_line * 2 + 1), dtype=float)
            gt_lane_type = np.zeros(shape=(self.feature_size, self.class_num), dtype=float)
            gt_lane_type[:, 0] = 1
            gt_loc = gt_lane_offset.astype(np.float32)
            gt_type = gt_lane_type.astype(np.float32)
        else:
            lane_set = trans_to_lane_with_type(gt_lanes_list)
            all_anchor_count = np.zeros(shape=(self.feature_height, self.feature_width))
            all_anchor_distance = list()
            all_anchor_loc = list()
            all_anchor_list = list()

            for lane in lane_set:
                cur_line = lane.lane
                new_lane = delete_repeat_y(cur_line)
                if len(new_lane) < 2:
                    startpos = -1
                    endpos = -1
                    x_list = []
                    y_list = []
                else:
                    interp_lane = spline_interp(lane=new_lane, step_t=1)
                    x_pt_list, y_pt_list = delete_nearby_point(interp_lane)
                    x_pt_list = x_pt_list[::-1]
                    y_pt_list = y_pt_list[::-1]
                    startpos, endpos, x_list, y_list = \
                        self.uniform_sample_lane_y_axis(x_pt_list, y_pt_list)
                if startpos == -1 or endpos == -1:
                    continue
                anchor_list, anchor_distance_result, gt_loc_list = \
                    self.get_one_line_pass_anchors(startpos, endpos, x_list, y_list, all_anchor_count)

                all_anchor_distance.append(anchor_distance_result)
                all_anchor_loc.append(gt_loc_list)
                all_anchor_list.append(anchor_list)

            if self.anchor_lane_num == 1:
                gt_type, gt_loc = self.get_one_lane_gt_loc_type(all_anchor_distance,
                                                                all_anchor_loc, all_anchor_count)
            elif self.anchor_lane_num == 2:
                gt_type, gt_loc = self.get_two_lane_gt_loc_type(all_anchor_distance,
                                                                all_anchor_loc, all_anchor_count)

        return gt_type, gt_loc

    def decode_lane(self, predict_type, predict_loc, cls_thresh):
        """Decode lane to normal type.

        :param predict_type: class result of groundtruth
        :type predict_type: nd.array whose shape is [576, class_num]
        :param predict_loc: regression result of groundtruth
        :type predict_loc: nd.array whose shape is [576, 145]=[576, 72+1+72]
        :return: lane set
        :rtype: dict
        """
        lane_set = list()
        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                prob = predict_type[index][1]
                if prob < cls_thresh:
                    continue
                down_anchor_lane = predict_loc[index, :self.points_per_line]
                up_anchor_lane = predict_loc[index, self.points_per_line:]
                relative_end_pos = up_anchor_lane[0]
                anchor_y_pos = int((self.feature_height - 1 - h) * self.points_per_anchor)
                anchor_center_x = (1.0 * w + 0.5) * self.step_w
                anchor_center_y = (1.0 * h + 0.5) * self.step_h
                up_lane = np.array([])
                down_lane = np.array([])
                end_pos = anchor_y_pos
                start_pos = anchor_y_pos
                for i in range(self.points_per_line):
                    if i >= relative_end_pos or anchor_y_pos + i >= self.points_per_line:
                        break
                    rela_x = up_anchor_lane[1 + i]
                    abs_x = anchor_center_x + rela_x
                    abs_y = self.input_height - 1 - (anchor_y_pos + i) * self.interval
                    p = Point(abs_x, abs_y)
                    up_lane = np.append(up_lane, p)
                    end_pos = anchor_y_pos + i + 1
                for i in range(anchor_y_pos):
                    rela_x = down_anchor_lane[i]
                    abs_x = anchor_center_x + rela_x
                    abs_y = self.input_height - 1 - (anchor_y_pos - 1 - i) * self.interval
                    p = Point(abs_x, abs_y)
                    down_lane = np.append(p, down_lane)
                    start_pos = anchor_y_pos - 1 - i
                if up_lane.size + down_lane.size >= 2:
                    lane = np.append(down_lane, up_lane)
                    lane_predict = Lane(prob, start_pos, end_pos,
                                        anchor_center_x, anchor_center_y, 1, lane)
                    lane_set.append(lane_predict)

        return lane_set

    def get_one_lane_gt_loc_type(self, all_anchor_distance, all_anchor_loc, all_anchor_count):
        """Get the location and type of one lane.

        :param all_anchor_distance: all anchors with distance
        :type all_anchor_distance: list of tuple
        :param all_anchor_loc: all anchor with correspond lane regression struct.
        :type all_anchor_loc: list
        :param all_anchor_count: the mask of weather anchor hit the lane or not.
        :type all_anchor_count: list
        :return gt_type: the type of groundtruth
        :rtype gt_type: nd.array
        :return gt_loc: the regression of groundtruth
        :rtype gt_loc: nd.array
        """
        gt_lane_offset = np.zeros(shape=(self.feature_size, self.pt_nums_single_lane), dtype=float)
        gt_lane_type = np.zeros(shape=(self.feature_size, self.class_num), dtype=float)
        gt_lane_type[:, 0] = 1

        for h in range(self.feature_height):
            for w in range(self.feature_width):
                index = h * self.feature_width + w
                cnt = all_anchor_count[h][w]
                gt_loc_list, gt_dist_list = \
                    get_lane_loc_list(all_anchor_distance, all_anchor_loc, h, w)

                if cnt == 0:
                    gt_lane_type[index, 0] = 1
                elif cnt == 1:
                    gt_lane_type[index, 0] = 0
                    gt_lane_type[index, 1] = 1
                    gt_lane_offset[index, :self.pt_nums_single_lane] = gt_loc_list[0]
                else:
                    gt_lane_type[index, 0] = 0
                    gt_lane_type[index, 1] = 1
                    line_loc_num = len(gt_loc_list)
                    line_dist_num = len(gt_dist_list)
                    if line_dist_num == line_loc_num:
                        [top_idx] = gettopk_idx(gt_dist_list)
                        gt_lane_offset[index, :self.pt_nums_single_lane] = gt_loc_list[top_idx]
                    else:
                        raise ValueError('Feature is Wrong.')

        gt_loc = gt_lane_offset.astype(np.float32)
        gt_type = gt_lane_type.astype(np.float32)

        return gt_type, gt_loc

    def uniform_sample_lane_y_axis(self, x_pt_list, y_pt_list):
        """Ensure y from bottom of image."""
        if len(x_pt_list) < 2 or len(y_pt_list) < 2:
            return -1, -1, [], []
        max_y = y_pt_list[-1]
        if max_y < self.input_height - 1:
            y1 = y_pt_list[-2]
            y2 = y_pt_list[-1]
            x1 = x_pt_list[-2]
            x2 = x_pt_list[-1]

            while max_y < self.input_height - 1:
                y_new = max_y + self.interval
                x_new = x1 + (x2 - x1) * (y_new - y1) / (y2 - y1)
                x_pt_list.append(x_new)
                y_pt_list.append(y_new)
                max_y = y_new

        x_list = np.array(x_pt_list)
        y_list = np.array(y_pt_list)
        if y_list.max() - y_list.min() < 5:
            return -1, -1, [], []
        if len(y_list) < 4:
            tck = interpolate.splrep(y_list, x_list, k=1, s=0)
        else:
            tck = interpolate.splrep(y_list, x_list, k=3, s=0)
        startpos = 0
        endpos = int((self.input_height - y_list[0]) / self.interval)
        if endpos > self.points_per_line - 1:
            endpos = self.points_per_line - 1
        if startpos >= endpos:
            return -1, -1, [], []

        y_list = []
        expand_pos = endpos
        for i in range(startpos, expand_pos + 1):
            y_list.append(self.input_height - 1 - i * self.interval)
        xlist = interpolate.splev(y_list, tck, der=0)

        for i in range(len(xlist)):
            if xlist[i] == 0:
                xlist[i] += 0.01

        return startpos, endpos, xlist, y_list

    def get_one_line_pass_anchors(self, startpos, endpos, xlist, y_list, anchor_count):
        """Get one line pass all anchors."""
        anchor_list = []
        anchor_distance_result = []
        Gt_loc_list = []

        for i in range(0, endpos - startpos + 1):
            h = self.feature_height - 1 - int((startpos + i) * self.interval / self.step_h)
            w = int(xlist[i] / self.step_w)
            if h < 0 or h > self.feature_height - 1 or w < 0 or w > self.feature_width - 1:
                continue
            if (h, w) in anchor_list:
                continue
            anchor_y = (1.0 * h + 0.5) * self.step_h
            center_x = (1.0 * w + 0.5) * self.step_w

            curr_y = self.input_height - 1 - i * self.interval
            if curr_y <= anchor_y:
                continue

            anchor_list.append((h, w))
            center_y = y_list[int(self.points_per_line / self.feature_height) * (self.feature_height - 1 - h)]

            loss_line = [0] * (self.points_per_line * 2 + 1)
            length = endpos - startpos + 1
            up_index = 0
            for j in range(0, length):
                if y_list[startpos + j] <= center_y:
                    loss_line[self.points_per_line + 1 + up_index] = xlist[j] - center_x
                    up_index += 1
            loss_line[self.points_per_line] = up_index
            down_index = length - up_index - 1
            for j in range(0, endpos - startpos + 1):
                if y_list[startpos + j] > center_y:
                    if xlist[j] - center_x == 0:
                        loss_line[down_index] = 0.000001
                    else:
                        loss_line[down_index] = xlist[j] - center_x
                    down_index -= 1

            Gt_loc_list.append(loss_line)
            anchor_count[h][w] += 1
            distance = xlist[i] - self.img_center_x
            anchor_distance_result.append((h, w, distance))

        return anchor_list, anchor_distance_result, Gt_loc_list
