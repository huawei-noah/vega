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

import json
import cv2
import numpy as np


class Point:
    """Point class represents and manipulates x,y coords.

    :param x: x coordinate
    :type x: float
    :param y: y coordinate
    :type y: float
    """

    def __init__(self, x=0, y=0):
        """Create a new point at the origin."""
        self.x = x
        self.y = y

    def __repr__(self):
        """Represent current class."""
        return "{}, {}".format(self.x, self.y)


class Lane:
    """This is the class of Lane class represents lane and its params.

    :param prob: the probablity of a line.
    :type prob: str
    :param start_pos: the start position of the lane
    :type start_pos: float
    :param end_pos: the end position of the lane
    :type end_pos: float
    :param anchor_x: the x coordinate of the lane
    :type anchor_x: float
    :param anchor_y: the y coordinate of the lane
    :type anchor_y: float
    :param type: the type of a lane
    :type type: int
    :param lane: the content of the lane
    :type lane: nd.arrray
    """

    def __init__(self, prob=0, start_pos=0, end_pos=0,
                 anchor_x=0, anchor_y=0, type=0, lane=np.array([])):
        self.prob = prob
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.lane = lane
        self.idx = 0
        self.ax = anchor_x
        self.ay = anchor_y
        self.type = type

    def __lt__(self, other):
        """Compare the lane with other lane by prob."""
        return self.prob > other.prob


def calc_y_cross(p1, p2, y):
    """Given two points of a same line and a y coordinate, calculate x coordinate.

    :param p1: the first point of the line.
    :type p1: Point
    :param p2: the second point of the line.
    :type p2: Point
    :param y: the y coordinate
    :type: float
    :return: x coordinate
    :rtype: float
    """
    if abs(p1.y - p2.y) < 1e-6:
        return -1
    k = (p1.x - p2.x) / (p1.y - p2.y)
    b = p1.x - k * p1.y
    return k * y + b


class LaneWithCrossK:
    """This is the class of Lane class represents lane and its params.

    :param idx_in: the index of the lane.
    :type idx_in: int
    :param y_in: the y coordiante of the lane.
    :type y_in: float
    :param lane: the content of the lane
    :type lane: nd.arrray
    """

    def __init__(self, lane_=Lane(), idx_in=0, y_in=0):
        self.lane = lane_
        self.idx = idx_in
        self.y = y_in
        if lane_.lane[1].y < lane_.lane[0].y:
            self.k = (lane_.lane[1].x - lane_.lane[0].x) / (lane_.lane[1].y - lane_.lane[0].y)
            self.cross_x = calc_y_cross(lane_.lane[0], lane_.lane[1], y_in)
        elif lane_.lane[1].y > lane_.lane[0].y:
            self.k = (lane_.lane[-1].x - lane_.lane[-2].x) / (lane_.lane[-1].y - lane_.lane[-2].y)
            self.cross_x = calc_y_cross(lane_.lane[-2], lane_.lane[-1], y_in)
        else:
            self.k = 1000
            self.cross_x = calc_y_cross(lane_.lane[-2], lane_.lane[-1], y_in)

    def __lt__(self, other):
        """Compare with other lane by x coordinate."""
        eps = 2.0
        x1_first = self.cross_x
        x2_first = other.cross_x
        if abs(x1_first - x2_first) > eps:
            return x1_first < x2_first
        else:
            if self.lane.lane[1].y < self.lane.lane[0].y:
                x1_last = self.lane.lane[-1].x
                x2_last = other.lane.lane[-1].x
            else:
                x1_last = self.lane.lane[0].x
                x2_last = other.lane.lane[0].x
            return x1_last < x2_last


def get_lane_list(lane_object, scale_x, scale_y):
    """Scale lane by x ratio and y ratio.

    :param lane_object: source lanes
    :type lane_object: dict
    :param scale_x: scale at x coordiante
    :type scale_x: float
    :param scale_y: scale at y coordiante
    :type scale_y: float
    :return: list of points list
    :rtype: list
    """
    gt_lane_list = []
    for line in lane_object["Lines"]:
        one_line = []
        set_y = []
        for point_index in range(len(line)):
            if line[point_index]["x"] == "nan" or line[point_index]["y"] == "nan":
                continue
            if not line[point_index]["y"] in set_y:
                set_y.append(line[point_index]["y"])
                one_line.append((float(line[point_index]["x"]) * scale_x,
                                 float(line[point_index]["y"]) * scale_y))
        if len(one_line) >= 2:
            if one_line[0][1] < one_line[1][1]:
                one_line = one_line[::-1]
            gt_lane_list.append(one_line)
    return gt_lane_list


def trans_to_lane_with_type(gt_lanes_list):
    """Convert point list to lane list.

    :param gt_lanes_list: a list of points list
    :type gt_lanes_list: list
    :return: a list of lane
    :rtype: list
    """
    lane_set = list()
    cnt = 1
    for lane in gt_lanes_list:
        new_lane = np.array([])
        for pt in lane:
            p = Point(pt[0], pt[1])
            new_lane = np.append(new_lane, p)
        if new_lane.size >= 2:
            prob = 1
            type = cnt
            c_x = 0
            c_y = 0
            start_pos = 0
            end_pos = 0
            _lane = Lane(prob, start_pos, end_pos, c_x, c_y, type, new_lane)
            lane_set.append(_lane)
            cnt += 1
    return lane_set


def order_lane_x_axis(lane_set, h):
    """Order lane by x axis.

    :param lane_set: a set of lanes
    :type lane_set: list
    :param h: y coordinate by which to compare the x coordiante
    :type h: float
    :return: orderd lanes
    :rtype: list
    """
    if len(lane_set) == 0:
        return list()
    cross_y = h - 1.0
    lanes_crossk = list()
    for i in range(len(lane_set)):
        lane_with_cross_k = LaneWithCrossK(lane_set[i], i, cross_y)
        lanes_crossk.append(lane_with_cross_k)

    lanes_crossk_sorted = sorted(lanes_crossk)

    right_pos = len(lanes_crossk_sorted)
    for i in range(len(lanes_crossk_sorted)):
        if lanes_crossk_sorted[i].k > 0:
            right_pos = i
            break

    lane_idx = [None] * len(lanes_crossk_sorted)
    idx = -1
    for i in range(right_pos - 1, -1, -1):
        lane_idx[i] = idx
        idx -= 1
    idx = 1
    for i in range(right_pos, len(lanes_crossk_sorted), 1):
        lane_idx[i] = idx
        idx += 1

    lanes_final = list()
    for i in range(len(lanes_crossk_sorted)):
        lanes_crossk_sorted[i].lane.idx = lane_idx[i]
        lanes_final.append(lanes_crossk_sorted[i].lane)

    return lanes_final


def get_lane_mean_x(lane):
    """Get mean x of the lane.

    :param lane: target lane
    :type lane: list
    :return: the mean value of x
    :rtype: float
    """
    x_pos_list = []
    for pt in lane:
        x_pos_list.append(pt.x)
    x_mean = 0
    if len(x_pos_list) == 1:
        x_mean = x_pos_list[0]
    elif len(x_pos_list) > 1:
        x_mean = (x_pos_list[0] + x_pos_list[-1]) / 2
    return x_mean


def convert_lane_to_dict(lane_set, sx, sy):
    """Convert lane to dict.

    :param lane_set: lane_set
    :type lane_set: dict
    :param sx: x ratio
    :type sx: float
    :param sy: y ratio
    :type sy: float
    :return: lane_dict
    :rtype: dict
    """
    list_dict = []
    for i in range(len(lane_set)):
        if lane_set[i].prob < 0.01:
            continue
        single_line = {}
        single_list = []
        for j in range(len(lane_set[i].lane)):
            p = lane_set[i].lane[j]
            single_list.append({'x': p.x * sx, 'y': p.y * sy})
        single_line['score'] = str(lane_set[i].prob)
        single_line['points'] = single_list
        list_dict.append(single_line)
    lane_dict = {'Lines': list_dict}
    return lane_dict


def save_dict_to_json(output_dict, output_file):
    """Save result dict to json.

    :param output_dict: the predict result dict
    :type output_dict: dict
    :param output_file: output file name
    :type output_file: str
    """
    jsonData = json.dumps(output_dict, separators=(',', ':'), indent=2)
    with open(output_file, 'w') as f:
        json.dump(json.loads(jsonData), f, sort_keys=True, separators=(',', ':'), indent=2)


def delete_repeat_y(cur_line):
    """Avoid same y with multi x.

    :param cur_line: the raw line
    :type cur_line:list
    :return: the deduplicated line
    :rtype:list
    """
    list_x = []
    list_y = []
    for pt in cur_line:
        list_x.append(pt.x)
        list_y.append(pt.y)

    sorted_y = sorted(list_y)
    sorted_x = []
    for i in range(len(sorted_y)):
        sorted_x.append(list_x[list_y.index(sorted_y[i])])

    set_sorted_y = []
    set_sorted_x = []
    index = 0
    for i in sorted_y:
        if not (i in set_sorted_y):
            set_sorted_y.append(i)
            set_sorted_x.append(sorted_x[index])
        index += 1

    new_lane = []
    if len(set_sorted_y) < 2:
        return new_lane

    for i in range(len(set_sorted_y)):
        new_lane.append({"x": set_sorted_x[i], "y": set_sorted_y[i]})
    if new_lane[0]["y"] < new_lane[1]["y"]:
        new_lane = new_lane[::-1]

    return new_lane


def trans_to_pt_list(interp_lane):
    """Interp lane to x list and y list.

    :param interp_lane: the raw line
    :type interp_lane:list
    :return:  x list and y list
    :rtype:list
    """
    x_pt_list = []
    y_pt_list = []
    for pt in interp_lane:
        cur_x = pt['x']
        cur_y = pt['y']
        x_pt_list.append(cur_x)
        y_pt_list.append(cur_y)
    return x_pt_list, y_pt_list


def delete_nearby_point(interp_lane):
    """Avoid too close of two lines.

    :param interp_lane: the raw line
    :type interp_lane:list
    :return: the processed line
    :rtype:list
    """
    x_pt_list = []
    y_pt_list = []
    cnt = 0
    pre_x = cur_x = 0
    pre_y = cur_y = 0

    for pt in interp_lane:
        if cnt == 0:
            pre_x = pt['x']
            pre_y = pt['y']
            x_pt_list.append(pre_x)
            y_pt_list.append(pre_y)
        else:
            cur_x = pt['x']
            cur_y = pt['y']
            if pre_y - cur_y < 1:
                continue
            else:
                x_pt_list.append(cur_x)
                y_pt_list.append(cur_y)
                pre_x = cur_x
                pre_y = cur_y
        cnt += 1

    return x_pt_list, y_pt_list


def get_lane_loc_list(all_anchor_distance, all_anchor_loc, h, w):
    """Get lane location list.

    :param all_anchor_distance: the distance of all anchor
    :type all_anchor_distance: list
    :param all_anchor_distance: the regression result of all anchor
    :type all_anchor_distance: list
    :param h: feature map height
    :type h: int
    :param w: feature map width
    :type w: int
    :return: gt loc list and gt dist list
    :rtype: list
    """
    gt_loc_list = list()
    gt_dist_list = list()
    lane_num = len(all_anchor_distance)
    for i in range(lane_num):
        cur_lane_anchor_list = all_anchor_distance[i]
        cur_lane_loc_list = all_anchor_loc[i]
        anchor_num = len(cur_lane_anchor_list)
        for j in range(anchor_num):
            cur_anchor = cur_lane_anchor_list[j]
            cur_anchor_loc = cur_lane_loc_list[j]
            cur_h = cur_anchor[0]
            cur_w = cur_anchor[1]
            if cur_h == h and cur_w == w:
                gt_loc_list.append(cur_anchor_loc)
                gt_dist_list.append(cur_anchor)
    return gt_loc_list, gt_dist_list


def gettopk_idx(gt_dist_list):
    """Get top index of the distance list.

    :param gt_dist_list: distance list
    :type gt_dist_list: list
    :return: index
    :rtype: int
    """
    distance_list = list()
    for cur_value in gt_dist_list:
        cur_distance = cur_value[2]
        distance_list.append(cur_distance)

    top_idx = np.argsort(distance_list)[:1]
    return top_idx


def draw_lane_set(org_img, lane_set, x_ratio=1, y_ratio=1, color=(0, 0, 255)):
    """Draw Multi line to an image.

    :param org_img: the image to be drawn
    :type org_img: nd.array
    :param lane_set: the set of lanes
    :type: list
    :param x_ratio: scale ratio of x axis
    :type: float
    :param y_ratio: scale ratio of x axis
    :type: float
    """
    for lane in lane_set:
        lane_points = lane.lane
        draw_single_line(org_img, lane_points, x_ratio, y_ratio, color)


def draw_single_line(org_img, lane_points, x_ratio=1, y_ratio=1, color=(0, 0, 255)):
    """Draw single line to an image.

    :param org_img: the image to be drawn
    :type org_img: nd.array
    :param lane_points: the points of the lane
    :type: list
    :param x_ratio: scale ratio of x axis
    :type: float
    :param y_ratio: scale ratio of x axis
    :type: float
    """
    for i in range(len(lane_points) - 1):
        p0 = lane_points[i]
        p1 = lane_points[i + 1]
        p0_x = int(p0.x)
        p0_y = int(p0.y)
        p1_x = int(p1.x)
        p1_y = int(p1.y)
        cv2.circle(org_img, (int(p0_x * x_ratio), int(p0_y * y_ratio)), 1, color=color, thickness=5)
        cv2.line(org_img, (int(p0_x * x_ratio), int(p0_y * y_ratio)),
                 (int(p1_x * x_ratio), int(p1_y * y_ratio)), color, 1)


def calc_err_dis_with_pos(l1, l2):
    """Calculate the distance of two lanes.

    :param l1: l1
    :type l1: Lane
    :param l2: l2
    :type l2: Lane
    :return: the distance of two lanes.
    :rtype: float
    """
    max_start_pos = max(l1.start_pos, l2.start_pos)
    min_end_pos = min(l1.end_pos, l2.end_pos)
    if min_end_pos <= max_start_pos or max_start_pos < 0 or min_end_pos < 1:
        return 10e6
    pts1 = l1.lane
    pts2 = l2.lane
    dis = 0.0
    for i in range(max_start_pos, min_end_pos):
        dis += abs(pts1[i - l1.start_pos].x - pts2[i - l2.start_pos].x)
    dis /= (min_end_pos - max_start_pos)
    dis_start = abs(l1.lane[max_start_pos - l1.start_pos].x - l2.lane[max_start_pos - l2.start_pos].x)
    dis = max(dis, dis_start)
    dis_end = abs(l1.lane[min_end_pos - 1 - l1.start_pos].x - l2.lane[min_end_pos - 1 - l2.start_pos].x)
    dis = max(dis, dis_end)
    return dis


def nms_with_pos(lane_set, thresh):
    """Execute Non Maximum Suppression for all lanes.

    :param lane_set: the set of all lanes as is the direct output of nn.
    :type lane_set: dict
    :param thresh: the distance thresh of every two lanes.
    :type thresh: float
    :return:the lanes after Non Maximum Suppression.
    :rtype:dict
    """
    if len(lane_set) == 0:
        return list()
    lane_sorted = sorted(lane_set)
    lanes_result = list()
    selected = [False] * len(lane_sorted)
    for n in range(len(lane_sorted)):
        if selected[n]:
            continue
        lanes_result.append(lane_sorted[n])
        selected[n] = True
        for t in range(n + 1, len(lane_sorted)):
            dis = calc_err_dis_with_pos(lane_sorted[n], lane_sorted[t])
            if dis <= thresh:
                selected[t] = True
    return lanes_result
