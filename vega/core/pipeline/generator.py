# -*- coding:utf-8 -*-

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

"""Generator for SearchPipeStep."""
import logging
import os
from pickle import HIGHEST_PROTOCOL
from copy import deepcopy
import vega
from vega.core.search_algs import SearchAlgorithm
from vega.core.search_space.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig
from vega.common.general import General
from vega.report import ReportServer, ReportClient
from vega.common.config import Config
from vega.common import update_dict
from vega.common.utils import remove_np_value
from vega.common.parameter_sharing import ParameterSharing
from vega.common import FileOps, TaskOps


class Generator(object):
    """Convert search space and search algorithm, sample a new model."""

    def __init__(self):
        self.step_name = General.step_name
        self.search_space = SearchSpace()
        self.search_alg = SearchAlgorithm(self.search_space)
        if hasattr(self.search_alg.config, 'objective_keys'):
            self.objective_keys = self.search_alg.config.objective_keys

    @property
    def is_completed(self):
        """Define a property to determine search algorithm is completed."""
        return self.search_alg.is_completed or vega.get_quota().quota_reached

    def sample(self):
        """Sample a work id and model from search algorithm."""
        out = []
        kwargs_list = []
        num_samples = 1
        for _ in range(10):
            res = self.search_alg.search()
            if not res:
                return None
            if not isinstance(res, list):
                res = [res]
            num_samples = len(res)
            if num_samples == 0:
                return None

            for sample in res:
                decode_sample = self.search_alg.decode(sample) if hasattr(
                    self.search_alg, "decode") else self._get_hps_desc_from_sample(sample)
                (worker_id, desc, hps, kwargs) = decode_sample + (dict(), ) * (4 - len(decode_sample))
                if not vega.get_quota().verify_sample(desc) or not vega.get_quota().verify_affinity(desc):
                    continue
                out.append((worker_id, desc, hps))
                kwargs_list.append(kwargs)
            if len(out) >= num_samples:
                break
        for i in range(num_samples):
            ReportClient().update(General.step_name, out[i][0], desc=out[i][1], hps=out[i][2], **kwargs_list[i])
        return out[:num_samples]

    def _get_hps_desc_from_sample(self, sample):
        if isinstance(sample, dict):
            worker_id = sample["worker_id"]
            desc = sample["encoded_desc"]
            sample.pop("worker_id")
            sample.pop("encoded_desc")
            kwargs = sample
            sample = _split_sample((worker_id, desc))
        else:
            kwargs = {}
            sample = _split_sample(sample)
        if hasattr(self, "objective_keys") and self.objective_keys:
            kwargs["objective_keys"] = self.objective_keys
        (worker_id, desc, hps) = sample
        if hasattr(self.search_alg.search_space, "to_desc"):
            desc = self.search_alg.search_space.to_desc(desc)
        elif desc.get("type") == 'DagNetwork':
            desc = desc
        else:
            desc = self._decode_hps(desc)
            hps = self._decode_hps(hps)
            network_desc = None
            if "modules" in desc:
                PipeStepConfig.model.model_desc = deepcopy(desc)
            elif "network" in desc:
                origin_desc = PipeStepConfig.model.model_desc
                network_desc = update_dict(desc["network"], origin_desc)
                PipeStepConfig.model.model_desc = network_desc
                desc.pop('network')

            (hps, desc) = self._split_hps_desc(hps, desc)
            if network_desc is not None:
                desc.update(network_desc)

        return worker_id, desc, hps, kwargs

    def _split_hps_desc(self, hps, desc):
        if "type" not in desc or desc.get("type") != "Sequential":
            del_items = []
            for item in desc:
                # TODO
                flag = item in ["modules", "networks",
                                "bit_candidates", "type", "nbit_a_list", "nbit_w_list",
                                "_arch_params"]
                flag = flag or ("modules" in desc and item in desc["modules"])
                if not flag:
                    hps[item] = desc[item]
                    del_items.append(item)
            for item in del_items:
                desc.pop(item)
        return hps, desc

    def update(self, step_name, worker_id):
        """Update search algorithm accord to the worker path.

        :param step_name: step name
        :param worker_id: current worker id
        :return:
        """
        record = ReportClient().get_record(step_name, worker_id)
        logging.debug("Get Record=%s", str(record))
        self.search_alg.update(record.serialize())
        ParameterSharing().remove()
        logging.info(f"Update Success. step_name={step_name}, worker_id={worker_id}")
        logging.info("Best values: %s", ReportServer().print_best(step_name=General.step_name))

    @staticmethod
    def _decode_hps(hps):
        """Decode hps: `trainer.optim.lr : 0.1` to dict format.

        And convert to `vega.common.config import Config` object
        This Config will be override in Trainer or Datasets class
        The override priority is: input hps > user configuration >  default configuration
        :param hps: hyper params
        :return: dict
        """
        hps_dict = {}
        if hps is None:
            return None
        if isinstance(hps, tuple):
            return hps
        for hp_name, value in hps.items():
            hp_dict = {}
            for key in list(reversed(hp_name.split('.'))):
                if hp_dict:
                    hp_dict = {key: hp_dict}
                else:
                    hp_dict = {key: value}
            hps_dict = update_dict(hps_dict, hp_dict, [])
        return Config(hps_dict)

    def dump(self):
        """Dump generator to file."""
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".generator")
        FileOps.dump_pickle(self, _file, protocol=HIGHEST_PROTOCOL)

    @classmethod
    def restore(cls):
        """Restore generator from file."""
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".generator")
        if os.path.exists(_file):
            return FileOps.load_pickle(_file)
        else:
            return None


def _split_sample(sample):
    """Split sample to (worker_id, model_desc, hps)."""
    if len(sample) not in [2, 3]:
        raise Exception("Incorrect sample length, sample: {}".format(sample))
    if len(sample) == 3:
        return sample[0], remove_np_value(sample[1]), remove_np_value(sample[2])
    if len(sample) == 2:
        mixed = deepcopy(sample[1])
        hps = {}
        for key in ["trainer", "dataset"]:
            if key in mixed:
                hps[key] = mixed[key]
                mixed.pop(key)
        return sample[0], remove_np_value(mixed), remove_np_value(hps)
