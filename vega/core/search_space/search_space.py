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

"""SearchSpace class."""
import logging
from collections import OrderedDict
from queue import Queue
import numpy as np
from vega.common.dag import DAG
from vega.common.class_factory import ClassFactory, ClassType
from vega.core.pipeline.conf import SearchSpaceConfig
from .param_types import PARAM_TYPE_MAP
from .condition_types import CONDITION_TYPE_MAP
from .params_factory import ParamsFactory
from .forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.SEARCHSPACE)
class SearchSpace(dict):
    """A search space for HyperParameter.

    :param hps: a dict contain HyperParameters, condition and forbidden.
    :type hps: dict, default is `None`.
    """

    def __init__(self, desc=None):
        """Init SearchSpace."""
        super(SearchSpace, self).__init__()
        if desc is None:
            desc = SearchSpaceConfig().to_dict()
            if desc.type is not None and desc.type != 'SearchSpace':
                cls = ClassFactory.get_cls(ClassType.SEARCHSPACE, desc.type)
                desc = cls.get_space(desc)
                if hasattr(cls, "to_desc"):
                    self.to_desc = cls.to_desc
        for name, item in desc.items():
            self.__setattr__(name, item)
            self.__setitem__(name, item)
        self._params = OrderedDict()
        self._condition_dict = OrderedDict()
        self._forbidden_list = []
        self._hp_count = 0
        self._dag = DAG()
        self.handler = None
        if desc is not None:
            self.form_desc(desc)

    @classmethod
    def get_space(self, desc):
        """Get Space."""
        return desc

    def form_desc(self, desc):
        """Create SearchSpace base on hyper-parameters object."""
        if 'hyperparameters' not in desc:
            return
        for space_dict in desc["hyperparameters"]:
            param = ParamsFactory.create_search_space(
                param_name=space_dict.get("key"),
                param_slice=space_dict.get('slice'),
                param_type=PARAM_TYPE_MAP[space_dict.get("type").upper()],
                param_range=space_dict.get("range"),
                generator=space_dict.get("generator"),
                sample_num=space_dict.get('sample_num')
            )
            self.add_hp(param)
        if "condition" in desc:
            for condition in desc["condition"]:
                _condition = ParamsFactory.create_condition(
                    self.get_hp(condition.get("child")),
                    self.get_hp(condition.get("parent")),
                    CONDITION_TYPE_MAP[condition.get("type").upper()],
                    condition.get("range")
                )
                self.add_condition(_condition)
        if "forbidden" in desc:
            for forbiddens in desc["forbidden"]:
                _forbiddens = []
                for _name, _value in forbiddens.items():
                    _forbiddens.append(ForbiddenEqualsClause(
                        param_name=self.get_hp(_name),
                        value=_value))
                self.add_forbidden_clause(
                    ForbiddenAndConjunction(_forbiddens))

    def sample(self):
        """Get the Sample of SearchSpace."""
        return self.decode(self.get_sample_space(1)[0])

    def verify_constraints(self, sample):
        """Verify condition."""
        for condition in self.get("condition", []):
            _type = condition["type"]
            child = condition["child"]  # eg. trainer.optimizer.params.momentum
            parent = condition["parent"]  # eg. trainer.optimizer.type
            _range = condition["range"]  # eg. range': ['SGD']
            if _type == "EQUAL" or _type == "IN":
                if parent in sample and sample[parent] in _range:
                    if child not in sample:
                        sample[child] = self.get_hp(child).sample()[0]
                elif child in sample:
                    del sample[child]
            if _type == "NOT_EQUAL":
                if parent in sample and sample[parent] in _range:
                    if child in sample:
                        del sample[child]
                elif child not in sample:
                    sample[child] = self.get_hp(child).sample()[0]
            # TODO condition type: IN, parent type: range
        return sample

    def size(self):
        """Get the size of SearchSpace, also the count of HyperParametera contained in this SearchSpace.

        :return: the size of SearchSpace.
        :rtype: int.

        """
        return self._hp_count

    def add_params(self, params):
        """Add params to the search space.

        :param list prams: List[HyperParameter].
        :return: List of added hyperparameters (same as input)
        :rtype: list

        """
        for param in params:
            if not ParamsFactory.is_params(param):
                raise TypeError("HyperParameter '%s' is not an instance of "
                                "SearchSpace.common.hyper_parameter."
                                "HyperParameter." % str(params))

        for param in params:
            self._add_hp(param)
        self._sort_hps()
        return self

    def add_hp(self, hyperparameter):
        """Add one hyperparameter to the hyperparameter space.

        :param HyperParameter hyperparameter: instance of `HyperParameter` to add.
        :return: hyperparameter (same as input)
        :rtype: HyperParameter

        """
        if not ParamsFactory.is_params(hyperparameter):
            raise TypeError("The method add_hp must be called "
                            "with an instance of SearchSpace."
                            "hyper_parameter.HyperParameter.")

        self._add_hp(hyperparameter)
        return self

    def _add_hp(self, hyperparameter):
        """Add one hyperparameter to the hyperparameter space.

        :param HyperParameter hyperparameter: instance of `HyperParameter` to add.

        """
        if hyperparameter.name in self._params:
            raise ValueError("HyperParameter `%s` is already in SearchSpace!"
                             % hyperparameter.name)
        self._params[hyperparameter.name] = hyperparameter
        self._hp_count = self._hp_count + 1
        self._dag.add_node(hyperparameter.name)

    def add_condition(self, condition):
        """Add new condition to the current SearchSpace.

        :param condition: `condition` that need to add.
        :type condition: instance of `Condition`.
        """
        if not ParamsFactory.is_condition(condition):
            raise ValueError('Not a valid condition {}'.format(condition))
        child_name = condition.child.name
        parent_name = condition.parent.name
        try:
            self._dag.add_edge(parent_name, child_name)
        except KeyError:
            raise KeyError('Hyperparameter in condition {} not exist in'
                           'current SearchSpace.'.format(condition))
        """
        except DAGValidationError:
            raise KeyError('Current condition {} valid DAG rule in current'
                           'SearchSpace, can not be added!'.format(condition))
        """
        if parent_name not in self._condition_dict:
            self._condition_dict[parent_name] = {}
        self._condition_dict[parent_name][child_name] = condition

    def add_forbidden_clause(self, forbidden_conjunction):
        """Add new ForbiddenAndConjunction to the current SearchSpace.

        :param forbidden_conjunction:  ForbiddenAndConjunction
        :type forbidden_conjunction: instance of `ForbiddenAndConjunction`.
        """
        if not isinstance(forbidden_conjunction, ForbiddenAndConjunction):
            raise ValueError(
                'Not a valid condition {}'.format(forbidden_conjunction))
        self._forbidden_list.append(forbidden_conjunction)

    def _sort_hps(self):
        """Sort the hyperparameter dictionary."""
        return

    def params(self):
        """Return the list of all hyperparameters.

        :return: List[HyperParameter]
        :rtype: list

        """
        return list(self._params.values())

    def get_hp_names(self):
        """Return the list of name of all hyperparameters.

        :return: List[str]
        :rtype: list


        """
        return list(self._params.keys())

    def get_hp(self, name):
        """Get HyperParameter by its name.

        :param str name: The name of HyperParameter.
        :return: HyperParameter
        :rtype: HyperParameter

        """
        hp = self._params.get(name)

        if hp is None:
            raise KeyError("HyperParameter '%s' does not exist in this "
                           "configuration space." % name)
        else:
            return hp

    def get_sample_space(self, n=1000, gridding=False):
        """Get the sampled param space from the current SearchSpace.

        :param int n: number of samples.
        :param bool gridding: use gridding sample or random sample.
        :return: shape is (n, len(self._hyperparameters)).
        :rtype: np.array

        """
        if gridding:
            return self._get_grid_sample_space()
        else:
            return self._get_random_sample_space(n)

    def _get_random_sample_space(self, n):
        """Get the sampled param space from the current SearchSpace.

        here we use the random sample, and return a np array of shape
        n*_hp_count, which is a sampled param space for GP or
        other model to predict.

        :param int n: sample count.
        :return: shape is (n, len(self._hyperparameters)).
        :rtype: np.array

        """
        parameters_array = np.zeros((n, self._hp_count))
        i = 0
        for _, hp in self._params.items():
            column = hp.sample(n=n, decode=False, handler=self.handler)
            parameters_array[:, i] = column
            i = i + 1
        return parameters_array

    def _generate_grid(self):
        """Get the all possible values for each of the tunables."""
        grid_axes = []
        for _, hp in self._params.items():
            grid_axes.append(hp.get_grid_axis(hp.slice))
        return grid_axes

    def _get_grid_sample_space(self):
        """Get the sampled param space from the current SearchSpace.

        here we use the random sample, and return a np array of shape
        n*len(_hyperparameters), which is a sampled param space for GP or
        other model to predict.

        :return: np.array, shape is (n, len(self._hyperparameters)).
        :rtype: np.array

        """
        param_list = [[]]
        params_grid = self._generate_grid()
        for param_grid in params_grid:
            param_list = [param_x + [param_y]
                          for param_x in param_list for param_y in param_grid]
        return param_list

    def decode(self, param_list):
        """Inverse transform a param list to original param dict.

        :param list param_list: the param list come from a search,
            in which params order are same with self._hyperparameters
        :return: the inverse transformed param dictionary.
        :rtype: dict

        """
        if len(param_list) != self._hp_count:
            raise ValueError("param_list length not equal to SearchSpace size!")
        i = 0
        assigned_forbidden_dict = {}
        inversed_param_dict = {}
        final_param_dict = {}
        for name, hp in self._params.items():
            param_value = param_list[i]

            forbidden_flag = False
            forbidden_value = []
            for forbidden_conjunction in self._forbidden_list:
                if name in forbidden_conjunction._forbidden_dict:
                    forbidden_flag = True

                    total_len = assigned_forbidden_dict.__len__() + forbidden_conjunction._forbidden_dict.__len__()
                    union_len = len(set(list(
                        assigned_forbidden_dict.items()
                    ) + list(forbidden_conjunction._forbidden_dict.items())))
                    # if assigned_forbidden_dict has same or similar forbidden conjunction
                    #  with `forbidden_conjunction`.
                    if (total_len - union_len) == \
                            forbidden_conjunction._forbidden_dict.__len__() - 1:
                        forbidden_value.append(
                            forbidden_conjunction._forbidden_dict.get(name))

            inversed_param_dict[name] = hp.decode(param_value, forbidden_value)
            if forbidden_flag:
                assigned_forbidden_dict[name] = inversed_param_dict[name]

            i = i + 1
        # check condition vaild
        # use DAG Breadth-First-Search to check each condition
        q = Queue()
        for ind_name in self._dag.ind_nodes():
            q.put(ind_name)
        while not q.empty():
            parent = q.get()
            final_param_dict[parent] = inversed_param_dict[parent]
            child_list = self._dag.next_nodes(parent)
            for child in child_list:
                condition = self._condition_dict[parent][child]
                if condition.evaluate(inversed_param_dict[parent]):
                    q.put(child)
        return final_param_dict


class SpaceSet(object):
    """Define a Space set to add search space dict."""

    def __init__(self, ):
        super(SpaceSet, self).__init__()
        self._search_space = []

    def add(self, key, space_type, space_range):
        """add one search space dict."""
        self._search_space.append({"key": key, "type": space_type, "range": space_range})
        return self

    def pop(self, idx):
        """Pop item by idx."""
        return self._search_space.pop(idx)

    def load(self, space_list):
        """Load search space list."""
        for space in space_list:
            if type(space) in [list, tuple]:
                self.add(*space)
            elif isinstance(space, dict):
                self.add(**space)
        return self.search_space

    @property
    def search_space(self):
        """Get all search spaces."""
        return SearchSpace(dict(hyperparameters=self._search_space))
