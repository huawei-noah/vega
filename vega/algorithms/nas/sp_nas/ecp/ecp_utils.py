# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Function of ecp evaluate."""
import mmcv
import os
import numpy as np
from ..spnet.ecp import ECPDataset
from .match import Evaluator, Result
from .params import ParamsFactory
from .get_iou import GetIou
from .lamr import lamr, create_lamr


def ecp_eval(results_path, det_path, gt_path, det_method_name, eval_type='pedestrian'):
    """Evaluate for ecp metrics.

    :param result_files: prediction folder path
    :type result_files: str
    :param det_path: prediction folder path
    :type det_path: str
    :param gt_path: gt folder path
    :type gt_path: str
    :param det_method_name: name
    :type det_method_name: str
    :param eval_type: type class for evaluation
    :type eval_type: str
    """
    print('Start evaluation for {}'.format(det_method_name))
    for difficulty in ['reasonable', 'small', 'occluded', 'all']:
        for ignore_other_vru in [True, False]:
            evaluate(difficulty, ignore_other_vru, results_path, det_path, gt_path, det_method_name,
                     use_cache=False, eval_type=eval_type)


def create_evaluator(data, difficulty, ignore_other_vru, type='pedestrian'):
    """Create evaluator.

    :param data: prediction
    :type data: list
    :param difficulty: 'reasonable', 'small', 'occluded', 'all'
    :type difficulty: str
    :param ignore_other_vru: gt folder path
    :type ignore_other_vru: bool
    :param type: type class for evaluation
    :type type: str
    """
    if type == 'pedestrian':
        params = ParamsFactory(difficulty=difficulty,
                               ignore_other_vru=ignore_other_vru,
                               tolerated_other_classes=['rider'],
                               dont_care_classes=['person-group-far-away'],
                               detections_type=['pedestrian'],
                               ignore_type_for_skipped_gts=1,
                               size_limits={'reasonable': 40, 'small': 30,
                                            'occluded': 40, 'all': 20},
                               occ_limits={'reasonable': 40, 'small': 40,
                                           'occluded': 80, 'all': 80},
                               size_upper_limits={'small': 60},
                               occ_lower_limits={'occluded': 40},
                               discard_depictions=True,
                               clipping_boxes=True,
                               transform_det_to_xy_coordinates=True
                               )
    elif type == 'rider':
        params = ParamsFactory(difficulty=difficulty,
                               ignore_other_vru=ignore_other_vru,
                               tolerated_other_classes=['pedestrian'],
                               dont_care_classes=['rider+vehicle-group-far-away'],
                               detections_type=['rider'],
                               ignore_type_for_skipped_gts=1,
                               size_limits={'reasonable': 40, 'small': 30,
                                            'occluded': 40, 'all': 20},
                               occ_limits={'reasonable': 40, 'small': 40,
                                           'occluded': 80, 'all': 80},
                               size_upper_limits={'small': 60},
                               occ_lower_limits={'occluded': 40},
                               discard_depictions=True,
                               clipping_boxes=True,
                               transform_det_to_xy_coordinates=True,
                               rider_boxes_including_vehicles=True
                               )
    else:
        assert False, 'Evaluation type not supported'

    return Evaluator(data,
                     metric=GetIou,
                     ignore_gt=params.ignore_gt,
                     skip_gt=params.skip_gt,
                     skip_det=params.skip_det,
                     preprocess_gt=params.preprocess_gt,
                     preprocess_det=params.preprocess_det,
                     )


def evaluate(difficulty, ignore_other_vru, results_path, det_path, gt_path, det_method_name,
             use_cache, eval_type='pedestrian'):
    """Evaluate to calculate lamr."""
    pkl_path = os.path.join(results_path,
                            'ignore={}_difficulty={}_evaltype={}.pkl'.format(ignore_other_vru,
                                                                             difficulty, eval_type))

    if os.path.exists(pkl_path) and use_cache:
        result = Result.load_from_disc(pkl_path)
    else:
        data = ECPDataset.load_gt_det(gt_path, det_path)
        evaluator = create_evaluator(data, difficulty, ignore_other_vru, eval_type)
        result = evaluator.result
        result.save_to_disc(pkl_path)

    mr = 1.0 - np.true_divide(result.tp, result.nof_gts)
    recall = np.true_divide(result.tp, result.nof_gts)
    fppi = np.true_divide(result.fp, result.nof_imgs)
    title = 'difficulty={}, ignore_other_vru={}, evaltype={}'.format(difficulty, ignore_other_vru,
                                                                     eval_type)
    label = 'lamr: {}'.format(lamr(recall, fppi))
    fig = create_lamr(title, label, mr, fppi)
    filename = 'lamr_ignore={}_difficulty={}_evaltype={}'.format(ignore_other_vru, difficulty,
                                                                 eval_type)
    fig.savefig(os.path.join(results_path, '{}.pdf'.format(filename)))
    fig.savefig(os.path.join(results_path, '{}.png'.format(filename)))
    return lamr(recall, fppi)


def results2frame(dataset, results, destdir):
    """Save results to frame.

    :param dataset: dataset
    :type dataset: dataset
    :param results: results
    :type results: list
    :param destdir: coco data class
    :type destdir: str
    """
    assert isinstance(results[0], list)
    for idx in range(len(dataset)):
        img_info = dataset.img_infos[idx]
        result = results[idx]
        children_list = []
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                _bbox = bboxes[i].tolist()
                data = dict()
                data['orient'] = -10.
                data['x0'] = _bbox[0]
                data['y0'] = _bbox[1]
                data['x1'] = _bbox[2]
                data['y1'] = _bbox[3]
                data['score'] = float(bboxes[i][4])
                data['identity'] = dataset.CLASSES[label]
                children_list.append(data)
        destfile = os.path.join(destdir, img_info['filename'].replace('.png', '.json'))
        if not os.path.exists('/'.join(destfile.split('/')[:-1])):
            os.makedirs('/'.join(destfile.split('/')[:-1]))
        mmcv.dump({'children': children_list, 'identity': "frame"}, destfile)
