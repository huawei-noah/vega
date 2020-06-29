# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""json_object_hooks.py."""
import numpy as np

from .common.condition_types import ConditionTypes
from .common.param_types import ParamTypes
from .hyperparameter_space import HyperParameter, HyperparameterSpace, Condition
from .common.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

PARAM_TYPE_MAP = {
    'INT': ParamTypes.INT,
    'INT_EXP': ParamTypes.INT_EXP,
    'INT_CAT': ParamTypes.INT_CAT,
    'FLOAT': ParamTypes.FLOAT,
    'FLOAT_EXP': ParamTypes.FLOAT_EXP,
    'FLOAT_CAT': ParamTypes.FLOAT_CAT,
    'STRING': ParamTypes.STRING,
    'BOOL': ParamTypes.BOOL
}

CONDITION_TYPE_MAP = {
    'EQUAL': ConditionTypes.EQUAL,
    'NOT_EQUAL': ConditionTypes.NOT_EQUAL,
    'IN': ConditionTypes.IN
}


def hp2json(dct):
    """Transform a params dict to json readable dict.

    :param dict dct: params dict.
    :return: json readable params dict.
    :rtype: dict

    """
    ret_dict = {}
    for key in dct:
        if isinstance(dct[key], np.int32) or isinstance(dct[key], np.int64):
            ret_dict[key] = int(dct[key])
        elif isinstance(dct[key], np.bool_):
            ret_dict[key] = bool(dct[key])
        elif isinstance(dct[key], np.float64):
            ret_dict[key] = float(dct[key])
        elif isinstance(dct[key], np.str_):
            ret_dict[key] = str(dct[key])
        else:
            ret_dict[key] = dct[key]
    return ret_dict


def json_to_hps(dct):
    """Translate the params dict to HyperparameterSpace object.

    :param dict dct: params dict.
    :return: HyperparameterSpace.
    :rtype: HyperparameterSpace or None
    """
    if "hyperparameters" in dct:
        hps = HyperparameterSpace()
        for hp_dict in dct["hyperparameters"]:
            hp_name = hp_dict.get("key")
            hp_slice = hp_dict.get('slice')
            hp_type = PARAM_TYPE_MAP[hp_dict.get("type").upper()]
            hp_range = hp_dict.get("range")
            hp = HyperParameter(param_name=hp_name, param_slice=hp_slice,
                                param_type=hp_type, param_range=hp_range)
            hps.add_hyperparameter(hp)
        if "condition" in dct:
            for cond_dict in dct["condition"]:
                cond_child = hps.get_hyperparameter(
                    cond_dict.get("child"))
                cond_parent = hps.get_hyperparameter(
                    cond_dict.get("parent"))
                cond_type = CONDITION_TYPE_MAP[cond_dict.get("type").upper()]
                cond_range = cond_dict.get("range")
                cond = Condition(cond_child, cond_parent, cond_type, cond_range)
                hps.add_condition(cond)
        if "forbidden" in dct:
            for forbidden_dict in dct["forbidden"]:
                forbidden_list = []
                for forb_name, forb_value in forbidden_dict.items():
                    forbidden_list.append(ForbiddenEqualsClause(
                        param_name=hps.get_hyperparameter(forb_name),
                        value=forb_value))
                hps.add_forbidden_clause(
                    ForbiddenAndConjunction(forbidden_list))
        return hps
    return None
