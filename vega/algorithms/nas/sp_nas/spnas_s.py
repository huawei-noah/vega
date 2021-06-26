"""The second stage of SMNAS."""
import copy
import logging
import random
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
import numpy as np
from .conf import SpNasConfig
from vega.report import ReportServer


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SpNasS(SearchAlgorithm):
    """Second Stage of SPNAS."""

    config = SpNasConfig()

    def __init__(self, search_space=None):
        super(SpNasS, self).__init__(search_space)
        self.need_reignite_queue = {}
        self.finished_reignite_queue = {}
        self.reignite_step_name = 'reignite'
        self.sample_count = 0
        self.max_sample = self.config.max_sample

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count > self.max_sample

    def _reignite(self, desc):
        reignite_desc = copy.deepcopy(self.config.reignite_desc)
        best_code = desc.get("network.backbone.code")
        reignite_desc['code'] = best_code
        self.need_reignite_queue[self.sample_count] = desc
        logging.info("Start to reignite work_id:{} desc:{}".format(self.sample_count, reignite_desc))
        return dict(worker_id=self.sample_count, encoded_desc=reignite_desc)

    def search(self):
        """Search a sample."""
        if self.need_reignite_queue:
            # share weights and use reignite model
            worker_id, desc = self.need_reignite_queue.popitem()
            reignite_record = ReportServer().get_record(self.step_name, worker_id)
            desc['network.backbone.code'] = reignite_record.desc['code']
            desc['network.backbone.weight_file'] = reignite_record.weights_file
            self.finished_reignite_queue[worker_id] = desc
            logging.info("Finished reignited models, work_id:{} desc:{}".format(worker_id, desc))
            return dict(worker_id=worker_id, encoded_desc=desc)
        pareto_records = ReportServer().get_pareto_front_records(choice='normal')
        best_record = pareto_records[0] if pareto_records else None
        desc = self.search_space.sample()
        arch_code = desc.get('network.backbone.code')
        if best_record:
            desc['network.backbone.weight_file'] = best_record.weights_file
        if best_record or not self.config.retain_original_code:
            desc['network.backbone.code'] = self._mutate_serialnet(arch_code)
        self.sample_count += 1
        if self.config.reignite and self.config.reignite_desc:
            return self._reignite(desc)
        logging.info("desc:{}".format(desc))
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    def update(self, record):
        """Update function, Not Implemented Yet.

        :param record: record dict.
        """
        logging.info("End to reignite work_id:{} performance:{}".format(record.get('worker_id'), record.get('rewards')))

    @property
    def max_samples(self):
        """Return the max number of samples."""
        return self.max_sample

    def _mutate_serialnet(self, arch):
        """Swap & Expend operation in Serial-level searching."""

        def is_valid(arch):
            stages = arch.split('-')
            for stage in stages:
                if len(stage) == 0:
                    return False
            return True

        def expend(arc):
            idx = np.random.randint(low=1, high=len(arc))
            arc = arc[:idx] + '1' + arc[idx:]
            return arc, idx

        def swap(arc, len_step=3):
            is_not_valid = True
            arc_origin = copy.deepcopy(arc)
            temp = arc.split('-')
            num_insert = len(temp) - 1
            while is_not_valid or arc == arc_origin:
                next_start = 0
                arc = list(''.join(temp))
                for i in range(num_insert):
                    pos = arc_origin[next_start:].find('-') + next_start
                    assert arc_origin[pos] == '-', "Wrong '-' is found!"
                    max_step = min(len_step, max(len(temp[i]), len(temp[i + 1])))
                    step_range = list(range(-1 * max_step, max_step))
                    step = random.choice(step_range)
                    next_start = pos + 1
                    pos = pos + step
                    arc.insert(pos, '-')
                arc = ''.join(arc)
                is_not_valid = (not is_valid(arc))
            return arc

        arch_origin = arch
        success = False
        k = 0
        while not success:
            k += 1
            arch = arch_origin
            ops = []
            for i in range(self.config.num_mutate):
                op_idx = np.random.randint(low=0, high=2)  # op_idx [0, 1]
                adds_thresh_ = self.config.addstage_ratio if len(arch.split('-')) < self.config.max_stages else 1
                if op_idx == 0 and random.random() > self.config.expend_ratio:
                    arch, idx = expend(arch)
                    arch, idx = expend(arch)
                elif op_idx == 1:
                    arch = swap(arch)
                elif op_idx == 2 and random.random() > adds_thresh_:
                    arch = arch + '-1'
                    ops.append('add stage')
                else:
                    ops.append('Do Nothing.')
            success = arch != arch_origin
            flag = 'Success' if success else 'Failed'
            logging.info('Serial-level Sample{}: {}. {}.'.format(k + 1, ' -> '.join(ops), flag))
        return arch
