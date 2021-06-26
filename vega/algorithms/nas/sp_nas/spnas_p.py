"""The second stage of SMNAS."""

import logging
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
import numpy as np
from .conf import SpNasConfig
from vega.report import ReportServer


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SpNasP(SearchAlgorithm):
    """Second Stage of SMNAS."""

    config = SpNasConfig()

    def __init__(self, search_space=None):
        super(SpNasP, self).__init__(search_space)
        self.sample_count = 0
        self.max_sample = self.config.max_sample

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search a sample."""
        pareto_records = ReportServer().get_pareto_front_records(choice='normal')
        best_record = pareto_records[0] if pareto_records else None
        desc = self.search_space.sample()
        if best_record:
            desc['network.neck.code'] = self._mutate_parallelnet(best_record.desc.get("neck").get('code'))
        self.sample_count += 1
        logging.info("desc:{}".format(desc))
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    @property
    def max_samples(self):
        """Return the max number of samples."""
        return self.max_sample

    def _mutate_parallelnet(self, code):
        """Mutate operation in Parallel-level searching.

        :param code: base arch encode
        :type code: list
        :return: parallel arch encode after mutate
        :rtype: list
        """
        p = [0.4, 0.3, 0.2, 0.1]
        num_stage = len(code)
        return list(np.random.choice(4, size=num_stage, replace=True, p=p))
