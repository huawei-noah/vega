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


def spline_interp(lane, step_t=1):
    """Interp target lane.

    :param lane: the lane without interp.
    :type lane: list of dict
    :param step_t: interp step
    :type step_t: float
    :return: the lane which is prrocessed
    :rtype: list of dict
    """
    interp_lane = []
    if len(lane) < 2:
        return lane
    interp_param = calc_params(lane)
    for f in interp_param:
        t = 0
        while t < f['h']:
            x = calc_x(f, t)
            y = calc_y(f, t)
            interp_lane.append({"x": x, "y": y})
            t += step_t
    interp_lane.append(lane[-1])
    return interp_lane


def calc_params(lane):
    """Calculate the interp params of target lane.

    :param lane: the lane to be calc
    :type lane: list
    :return: the params
    :rtype: dict
    """
    params = []
    n_pt = len(lane)
    if n_pt < 2:
        return params
    if n_pt == 2:
        h0 = np.sqrt((lane[0]['x'] - lane[1]['x']) * (lane[0]['x'] - lane[1]['x']) +  # noqa W504
                     (lane[0]['y'] - lane[1]['y']) * (lane[0]['y'] - lane[1]['y']))
        a_x = lane[0]['x']
        a_y = lane[0]['y']
        b_x = (lane[1]['x'] - a_x) / h0
        b_y = (lane[1]['y'] - a_y) / h0
        params.append({"a_x": a_x, "b_x": b_x, "c_x": 0, "d_x": 0, "a_y": a_y, "b_y": b_y, "c_y": 0, "d_y": 0, "h": h0})
        return params
    h = []
    for i in range(n_pt - 1):
        dx = lane[i]['x'] - lane[i + 1]['x']
        dy = lane[i]['y'] - lane[i + 1]['y']
        h.append(np.sqrt(dx * dx + dy * dy))
    A = []
    B = []
    C = []
    D_x = []
    D_y = []
    for i in range(n_pt - 2):
        A.append(h[i])
        B.append(2 * (h[i] + h[i + 1]))
        C.append(h[i + 1])
        dx1 = (lane[i + 1]['x'] - lane[i]['x']) / h[i]
        dx2 = (lane[i + 2]['x'] - lane[i + 1]['x']) / h[i + 1]
        tmpx = 6 * (dx2 - dx1)
        dy1 = (lane[i + 1]['y'] - lane[i]['y']) / h[i]
        dy2 = (lane[i + 2]['y'] - lane[i + 1]['y']) / h[i + 1]
        tmpy = 6 * (dy2 - dy1)
        if i == 0:
            C[i] /= B[i]
            D_x.append(tmpx / B[i])
            D_y.append(tmpy / B[i])
        else:
            base_v = B[i] - A[i] * C[i - 1]
            C[i] /= base_v
            D_x.append((tmpx - A[i] * D_x[i - 1]) / base_v)
            D_y.append((tmpy - A[i] * D_y[i - 1]) / base_v)

    Mx = np.zeros(n_pt)
    My = np.zeros(n_pt)
    Mx[n_pt - 2] = D_x[n_pt - 3]
    My[n_pt - 2] = D_y[n_pt - 3]
    for i in range(n_pt - 4, -1, -1):
        Mx[i + 1] = D_x[i] - C[i] * Mx[i + 2]
        My[i + 1] = D_y[i] - C[i] * My[i + 2]

    Mx[0] = 0
    Mx[-1] = 0
    My[0] = 0
    My[-1] = 0

    for i in range(n_pt - 1):
        a_x = lane[i]['x']
        b_x = (lane[i + 1]['x'] - lane[i]['x']) / h[i] - (2 * h[i] * Mx[i] + h[i] * Mx[i + 1]) / 6
        c_x = Mx[i] / 2
        d_x = (Mx[i + 1] - Mx[i]) / (6 * h[i])

        a_y = lane[i]['y']
        b_y = (lane[i + 1]['y'] - lane[i]['y']) / h[i] - (2 * h[i] * My[i] + h[i] * My[i + 1]) / 6
        c_y = My[i] / 2
        d_y = (My[i + 1] - My[i]) / (6 * h[i])

        params.append(
            {"a_x": a_x, "b_x": b_x, "c_x": c_x, "d_x": d_x, "a_y": a_y, "b_y": b_y, "c_y": c_y, "d_y": d_y, "h": h[i]})

    return params


def calc_x(f, t):
    """Calc x coordinate by params.

    :param f: the interp params
    :type f: dict
    :param t: the accumulation of steps
    :type t: int
    :return: x coordinate
    :rtype: float
    """
    return f['a_x'] + f['b_x'] * t + f['c_x'] * t * t + f['d_x'] * t * t * t


def calc_y(f, t):
    """Calc y coordinate by params.

    :param f: the interp params
    :type f: dict
    :param t: the accumulation of steps
    :type t: int
    :return: y coordinate
    :rtype: float
    """
    return f['a_y'] + f['b_y'] * t + f['c_y'] * t * t + f['d_y'] * t * t * t
