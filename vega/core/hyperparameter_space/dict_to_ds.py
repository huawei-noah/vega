# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""DiscreteSpaceBuilder for NAS random search."""
import queue
import copy
from .common.condition_types import ConditionTypes
from .common.param_types import ParamTypes
from .hyperparameter_space import HyperParameter, HyperparameterSpace, Condition
from .json_object_hooks import PARAM_TYPE_MAP


class DiscreteSpaceBuilder():
    """DiscreteSpaceBuilder.

    :param searchspace_cfg: Description of parameter `searchspace_cfg`.
    :type searchspace_cfg: type
    """

    def __init__(self, searchspace_cfg):
        """Init DiscreteSpaceBuilder."""
        self.hp_dict = {}
        self.cond_dict = {}
        if "modules" in searchspace_cfg:
            for key in searchspace_cfg['modules']:
                self.config_to_discrete_space(str(key), searchspace_cfg[key])
        else:
            for key, value in searchspace_cfg.items():
                if str(key) == 'name':
                    continue
                else:
                    self.config_to_discrete_space(str(key), value)
        self.discrete_space = HyperparameterSpace()
        for _, hp in self.hp_dict.items():
            self.discrete_space.add_hyperparameter(hp)
        for _, cond in self.cond_dict.items():
            self.discrete_space.add_condition(cond)

    def get_discrete_space(self):
        """Return a discrete search space defined by `searchspace_cfg`.

        :return: a discrete search space.
        :rtype: HyperparameterSpace
        """
        return self.discrete_space

    def config_to_discrete_space(self, name, config):
        """Transform a config to a discrete HyperparameterSpace.

        :param name: name
        :type name: str
        :param config: configs
        :type config: dict
        """
        d_queue = queue.Queue()
        tmp_dict = dict()
        tmp_dict[name] = config
        d_queue.put(tmp_dict)
        is_needed_parent = False
        while not d_queue.empty():
            tmp_dict = d_queue.get()
            for key, value in tmp_dict.items():
                config_dict = copy.deepcopy(value)
                child_hp = None
                if isinstance(config_dict, dict):
                    is_needed_parent, param_type, child_hp = self._dict_config_to_hps(
                        is_needed_parent, d_queue, key, config_dict)
                elif isinstance(config_dict, set):
                    is_needed_parent, param_type, child_hp = self._set_config_to_hps(
                        is_needed_parent, d_queue, key, config_dict)
                elif isinstance(config_dict, list):
                    is_needed_parent, param_type, child_hp = self._list_config_to_hps(
                        is_needed_parent, d_queue, key, config_dict)
                else:
                    is_needed_parent, param_type, child_hp = self._value_config_to_hps(
                        is_needed_parent, d_queue, key, config_dict)
                if child_hp is not None:
                    self.hp_dict[str(key)] = child_hp
                if is_needed_parent and child_hp is not None:
                    parent_name = ".".join(key.split(".")[:-2])
                    cond_value = key.split(".")[-2]
                    parent_hp = self.hp_dict[parent_name]
                    cond = Condition(child=child_hp, parent=parent_hp,
                                     condition_type=ConditionTypes.EQUAL,
                                     condition_range=[str(cond_value)])
                    self.cond_dict["{}.{}-cond".format(str(key), str(cond_value))] = cond

    def _dict_config_to_hps(self, is_needed_parent, d_queue, key, config_dict):
        """Extend to config_to_discrete_space.

        :param key: key
        :type key: key
        :param config_dict: config_dict
        :type config_dict: dict
        :return: param_type, child_hp.
        :rtype: param_type, child_hp
        """
        param_range = []
        param_type, child_hp = None, None
        for sub_key, _ in config_dict.items():
            param_range.append(str(sub_key))
        hp = HyperParameter(param_name=str(key), param_type=ParamTypes.STRING, param_range=param_range)
        add_this_parent = False
        for sub_key, sub_dict in config_dict.items():
            if isinstance(sub_dict, dict):
                is_needed_parent = True
                add_this_parent = True
        if add_this_parent:
            self.hp_dict[str(key)] = hp
        for sub_key, sub_dict in config_dict.items():
            if isinstance(config_dict, dict):
                sub_tmp_dict = {}
                sub_tmp_dict["{}.{}".format(key, sub_key)] = sub_dict
                d_queue.put(sub_tmp_dict)
        if add_this_parent and len(key.split(".")) >= 2:
            parent_name = ".".join(key.split(".")[:-1])
            cond_value = key.split(".")[-1]
            parent_hp = self.hp_dict[parent_name]
            cond = Condition(child=hp, parent=parent_hp,
                             condition_type=ConditionTypes.EQUAL,
                             condition_range=[str(cond_value)])
            self.cond_dict["{}.{}-cond".format(str(key), str(cond_value))] = cond
        return is_needed_parent, param_type, child_hp

    def _set_config_to_hps(self, is_needed_parent, d_queue, key, config_dict):
        """Extend to config_to_discrete_space.

        :param key: key
        :type key: key
        :param config_dict: config_dict
        :type config_dict: dict
        :return: param_type, child_hp.
        :rtype: param_type, child_hp
        """
        param_type = None
        child_hp = None
        if len(config_dict) != 2 or isinstance(config_dict[0], str):
            raise ValueError("{}'s is not like (TYPE, config_list)".format(config_dict))
        elif config_dict[0].upper() not in PARAM_TYPE_MAP:
            raise ValueError("{} type is not support".format(config_dict[0]))
        elif isinstance(config_dict[1], list):
            param_type = PARAM_TYPE_MAP[config_dict[0].upper()]
            child_hp = HyperParameter(param_name=key, param_type=param_type, param_range=config_dict)
        else:
            param_type = PARAM_TYPE_MAP[config_dict[0].upper()]
            child_hp = HyperParameter(param_name=key, param_type=param_type, param_range=[config_dict])
        return is_needed_parent, param_type, child_hp

    def _list_config_to_hps(self, is_needed_parent, d_queue, key, config_dict):
        """Extend to config_to_discrete_space.

        :param key: key
        :type key: key
        :param config_dict: config_dict
        :type config_dict: dict
        :return: param_type, child_hp.
        :rtype: param_type, child_hp
        """
        param_type = None
        child_hp = None
        if isinstance(config_dict[0], int):
            param_type = ParamTypes.INT_CAT
        elif isinstance(config_dict[0], float):
            param_type = ParamTypes.FLOAT_CAT
        elif isinstance(config_dict[0], str):
            param_type = ParamTypes.STRING
        elif isinstance(config_dict[0], bool):
            param_type = ParamTypes.BOOL
        else:
            raise ValueError("{}'s type is not in [int, float, bool, str]'".format(config_dict[0]))
        child_hp = HyperParameter(param_name=key, param_type=param_type, param_range=config_dict)
        return is_needed_parent, param_type, child_hp

    def _value_config_to_hps(self, is_needed_parent, d_queue, key, config_dict):
        """Extend to config_to_discrete_space.

        :param key: key
        :type key: key
        :param config_dict: config_dict
        :type config_dict: dict
        :return: param_type, child_hp.
        :rtype: param_type, child_hp
        """
        param_type = None
        child_hp = None
        if isinstance(config_dict, int):
            param_type = ParamTypes.INT_CAT
        elif isinstance(config_dict, float):
            param_type = ParamTypes.FLOAT_CAT
        elif isinstance(config_dict, str):
            param_type = ParamTypes.STRING
        elif isinstance(config_dict, bool):
            param_type = ParamTypes.BOOL
        else:
            raise ValueError("{}'s type is not in [int, float, bool, str]'".format(config_dict[0]))
        child_hp = HyperParameter(param_name=key, param_type=param_type, param_range=[config_dict])
        return is_needed_parent, param_type, child_hp

    def sample_to_dict(self, sample):
        """Sample to dict.

        :param sample: input sample
        :type sample: dict
        :return: return dict
        :rtype: dict
        """
        dict_sample = dict()
        sample_list = sorted(sample.items(), key=lambda x: len(x[0].split(".")), reverse=True)
        for key, value in sample_list:
            name_list = key.split(".")
            tmp_dict = dict_sample
            for i in range(len(name_list) - 1):
                if name_list[i] not in tmp_dict:
                    tmp_dict[name_list[i]] = {}
                tmp_dict = tmp_dict[name_list[i]]
            if name_list[-1] in tmp_dict:
                tmp_dict = tmp_dict[name_list[-1]]
                if isinstance(tmp_dict, dict):
                    if value not in tmp_dict:
                        tmp_dict[value] = {}
                else:
                    tmp_dict[name_list[-1]] = value
            else:
                tmp_dict[name_list[-1]] = value
        return dict_sample
