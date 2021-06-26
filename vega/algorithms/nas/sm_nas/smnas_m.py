"""The second stage of SMNAS."""

import logging
import random
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from vega.common import ConfigSerializable
from vega.networks.model_config import ModelConfig


class SMNasConfig(ConfigSerializable):
    """SR Config."""

    max_sample = 2
    min_sample = 1
    random_ratio = 0.2
    num_mutate = 10


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SMNasM(SearchAlgorithm):
    """Second Stage of SMNAS."""

    config = SMNasConfig()

    def __init__(self, search_space=None):
        super(SMNasM, self).__init__(search_space)
        # ea or random
        self.num_mutate = self.config.num_mutate
        self.random_ratio = self.config.random_ratio
        self.max_sample = self.config.max_sample
        self.min_sample = self.config.min_sample
        self.sample_count = 0
        logging.info("inited SMNasM")

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search a sample."""
        base_code = self.codec.encode(ModelConfig.model_desc)
        selected_code = self.ea_sample(base_code)
        desc = self.codec.decode(selected_code)
        self.sample_count += 1
        return self.sample_count, desc

    @property
    def max_samples(self):
        """Return the max number of samples."""
        return self.max_sample

    def insert(self, code):
        """Return new arch code."""
        idx = random.randint(0, len(code) - 1)
        code = list(code)
        code.insert(idx, '1')
        code = "".join(code)
        return code

    def remove(self, code):
        """Return new arch code."""
        ones_index = [i for i, char in enumerate(code) if char == '1']
        if not ones_index:
            return code
        idx = random.choice(ones_index)
        code = list(code)
        code.pop(idx)
        code = "".join(code)
        return code

    def swap(self, code):
        """Return new arch code."""
        not_ones_index = [i for i, char in enumerate(code) if char != '1']
        if len(not_ones_index) == 0:
            return code
        idx = random.choice(not_ones_index)
        index_list = [i for i in range(len(code))]
        index_list.pop(idx)
        swap_id = random.choice(index_list)
        code = list(code)
        code[idx], code[swap_id] = code[swap_id], code[idx]
        code = "".join(code)
        return code

    def ea_sample(self, code, num_mutate=3):
        """Use ea to sample a model."""
        code_list = code.split("-")
        new_code = []
        for cur_code in code_list:
            for i in range(num_mutate):
                op_idx = random.randint(0, 2)
                if op_idx == 0:
                    cur_code = self.insert(cur_code)
                elif op_idx == 1:
                    cur_code = self.remove(cur_code)
                elif op_idx == 2:
                    cur_code = self.swap(cur_code)
            new_code.append(cur_code)
        return "-".join(new_code)
