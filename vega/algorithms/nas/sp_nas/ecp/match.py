# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Process data and save results."""
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import numpy as np


class Result:
    """Class of save evaluation result."""

    def __init__(self, raw_result, skipped_gts=None, skipped_dets=None):
        self.raw_result = raw_result
        self.skipped_gts = skipped_gts
        self.skipped_dets = skipped_dets
        self.dets_including_ignored = []
        self.gts_including_ignored = []
        for frame in raw_result:
            self.dets_including_ignored.extend(frame['det']['children'])
            self.gts_including_ignored.extend(frame['gt']['children'])
        self.dets_including_ignored.sort(key=lambda k: k['score'], reverse=True)
        self.dets = [det for det in self.dets_including_ignored if det['matched'] != -1]
        self.gts = [gt for gt in self.gts_including_ignored if gt['ignore'] == 0]
        tp = np.array([(1 if det['matched'] == 1 else 0) for det in self.dets])
        fp = 1 - tp
        self.tp = np.cumsum(tp)
        self.fp = np.cumsum(fp)
        self.nof_gts = len(self.gts)
        self.nof_imgs = len(self.raw_result)

    def save_to_pkl(self, path):
        """Save results to pkl."""
        with open(path, 'wb') as f:
            pkl.dump([self.raw_result, self.skipped_gts, self.skipped_dets], f, protocol=-1)

    @classmethod
    def load_from_pkl(cls, path):
        """Load results from pkl."""
        with open(path, 'rb') as f:
            raw_result, skipped_gts, skipped_dets = pkl.load(f)
            return cls(raw_result, skipped_gts, skipped_dets)


class Evaluator:
    """Judgment data."""

    def __init__(self,
                 data,
                 metric,
                 comparable_identities=True,
                 ignore_gt=0,
                 skip_gt=False,
                 skip_det=False,
                 preprocess_gt=None,
                 preprocess_det=None,
                 allow_multiple_matches=False,
                 ):

        self.data = data
        self.metric = metric
        self.ignore_gt = ignore_gt
        self.skip_gt = skip_gt
        self.skip_det = skip_det
        self.preprocess_gt = preprocess_gt
        self.preprocess_det = preprocess_det
        self.comparable_identities = comparable_identities
        self.allow_multiple_matches = allow_multiple_matches
        self.skipped_gts = {'count': 0, 'types': set()}
        self.skipped_dets = {'count': 0, 'types': set()}
        self._current_frame = None
        self.__det_and_gt_id = 1
        self._raw_result = []
        self._run()
        self.result = Result(self._raw_result, skipped_gts=self.skipped_gts,
                             skipped_dets=self.skipped_dets)

    def _run(self):
        for frame in self.data:
            self._prepare_next_frame(frame)
            self._evaluate_current_frame()

    def _prepare_next_frame(self, frame):
        gts = []
        dets = []
        for gt in frame['gt']['children']:
            self.preprocess_gt(gt)
            if self.skip_gt(gt):
                self.skipped_gts['count'] += 1
                self.skipped_gts['types'].add(gt['identity'])
                continue
            gt['ignore'] = self.ignore_gt(gt)
            gt['matched'] = 0
            gt['__id__'] = self.__det_and_gt_id
            self.__det_and_gt_id += 1
            gts.append(gt)
        for det in frame['det']['children']:
            self.preprocess_det(det)
            if self.skip_det(det):
                self.skipped_dets['count'] += 1
                self.skipped_dets['types'].add(det['identity'])
                continue
            det['matched'] = 0
            det['__id__'] = self.__det_and_gt_id
            self.__det_and_gt_id += 1
            dets.append(det)
        frame['gt']['children'] = gts
        frame['det']['children'] = dets
        self._current_frame = frame

    def _evaluate_current_frame(self):
        gts = self._current_frame['gt']['children']
        dets = self._current_frame['det']['children']
        dets.sort(key=lambda k: k['score'], reverse=True)
        gts.sort(key=lambda k: k['ignore'])
        for det in dets:
            current_score = None
            idx_best_gt = -1
            matched_with_ignore = False
            for idx_gt, gt in enumerate(gts):
                if idx_best_gt >= 0 and not matched_with_ignore and gt['ignore']:
                    break
                if gt['matched'] and not self.allow_multiple_matches:
                    continue
                metric = self.metric(gt=gt, det=det)
                if not metric.match:
                    continue
                if not metric.better_match(current_score):
                    continue
                current_score = metric.iou
                idx_best_gt = idx_gt
                if gt['ignore']:
                    matched_with_ignore = True
            if idx_best_gt >= 0:
                det['metric_score'] = current_score
                if matched_with_ignore:
                    det['matched'] = -1
                else:
                    det['matched'] = 1
                det['matched_with'] = gts[idx_best_gt]['__id__']
                if not matched_with_ignore:
                    gts[idx_best_gt]['matched'] = 1
                    gts[idx_best_gt]['matched_by'] = det['__id__']
                    gts[idx_best_gt]['matched_by_score'] = det['score']
        self._raw_result.append(self._current_frame)
