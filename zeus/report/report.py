# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""
import json
import logging
import os
import glob
import traceback
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import pareto
from collections import OrderedDict
from zeus.common import FileOps, TaskOps
from zeus.common.general import General
from zeus.common import copy_search_file
from zeus.report.share_memory import ShareMemory
from .nsga_iii import SortAndSelectPopulation
from .record import ReportRecord


class Report(object):
    """Report class to save all records and broadcast records to share memory."""

    _hist_records = OrderedDict()
    REPORT_FILE_NAME = 'reports'
    BEST_FILE_NAME = 'best'
    __instances__ = None

    def __new__(cls, *args, **kwargs):
        """Override new method, singleton."""
        if not cls.__instances__:
            cls.__instances__ = super().__new__(cls, *args, **kwargs)
        return cls.__instances__

    def add(self, record):
        """Add one record into set."""
        self._hist_records[record.uid] = record

    @property
    def all_records(self):
        """Get all records."""
        return deepcopy(list(self._hist_records.values()))

    def print_best(self, step_name):
        """Print best performance and desc."""
        records = self.get_pareto_front_records(step_name)
        return [dict(worker_id=record.worker_id, performance=record._performance, desc=record.desc) for record in
                records]

    def pareto_front(self, step_name=None, nums=None, records=None):
        """Get parent front. pareto."""
        if records is None:
            records = self.all_records
            records = list(filter(lambda x: x.step_name == step_name and x.performance is not None, records))
        in_pareto = [record.rewards if isinstance(record.rewards, list) else [record.rewards] for record in records]
        if not in_pareto:
            return None, None
        try:
            fitness = np.array(in_pareto)
            if fitness.shape[1] != 1 and nums is not None and len(in_pareto) > nums:
                # len must larger than nums, otherwise dead loop
                _, res, selected = SortAndSelectPopulation(fitness.T, nums)
            else:
                outs = pareto.eps_sort(fitness, maximize_all=True, attribution=True)
                res, selected = np.array(outs)[:, :-2], np.array(outs)[:, -1].astype(np.int32)
            return res.tolist(), selected.tolist()
        except Exception as ex:
            logging.error('No pareto_front_records found, ex=%s', ex)
            return [], []

    def get_step_records(self, step_name=None):
        """Get step records."""
        if not step_name:
            step_name = General.step_name
        records = self.all_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps, records))
        return records

    def get_pareto_front_records(self, step_name=None, nums=None):
        """Get Pareto Front Records."""
        if not step_name:
            step_name = General.step_name
        records = self.all_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps and x.performance is not None, records))
        outs, selected = self.pareto_front(step_name, nums, records=records)
        if not outs:
            return []
        else:
            return [records[idx] for idx in selected]

    def dump_report(self, step_name=None, record=None):
        """Save one records."""
        try:
            if record and step_name:
                self._append_record_to_csv(self.REPORT_FILE_NAME, step_name, record.serialize())
            self.backup_output_path()

            step_path = TaskOps().step_path
            _file = os.path.join(step_path, ".reports")
            _dump_data = [Report._hist_records, Report.__instances__]
            with open(_file, "wb") as f:
                pickle.dump(_dump_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logging.warning(traceback.format_exc())

    @classmethod
    def restore(cls):
        """Transfer cvs_file to records."""
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".reports")
        if os.path.exists(_file):
            with open(_file, "rb") as f:
                data = pickle.load(f)
            cls._hist_records = data[0]
            cls.__instances__ = data[1]

    def backup_output_path(self):
        """Back up output to local path."""
        backup_path = TaskOps().backup_base_path
        if backup_path is None:
            return
        FileOps.copy_folder(TaskOps().local_output_path, backup_path)

    def output_pareto_front(self, step_name, desc=True, weights_file=False, performance=False):
        """Save one records."""
        logging.debug("All records in report, records={}".format(self.all_records))
        records = deepcopy(self.get_pareto_front_records(step_name))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump pareto front records, report is emplty.")
            return
        self._output_records(step_name, records, desc, weights_file, performance)

    def output_step_all_records(self, step_name, desc=True, weights_file=False, performance=False):
        """Output step all records."""
        records = self.all_records
        logging.debug("All records in report, records={}".format(self.all_records))
        records = list(filter(lambda x: x.step_name == step_name, records))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump records, report is emplty.")
            return
        self._output_records(step_name, records, desc, weights_file, performance)
        logging.info(self.print_best(step_name))

    def _output_records(self, step_name, records, desc=True, weights_file=False, performance=False):
        """Dump records."""
        columns = ["worker_id", "performance", "desc"]
        outputs = []
        for record in records:
            record = record.serialize()
            _record = {}
            for key in columns:
                _record[key] = record[key]
            outputs.append(deepcopy(_record))
        data = pd.DataFrame(outputs)
        step_path = FileOps.join_path(TaskOps().local_output_path, step_name)
        FileOps.make_dir(step_path)
        _file = FileOps.join_path(step_path, "output.csv")
        try:
            data.to_csv(_file, index=False)
        except Exception:
            logging.error("Failed to save output file, file={}".format(_file))
        for record in outputs:
            worker_id = record["worker_id"]
            worker_path = TaskOps().get_local_worker_path(step_name, worker_id)
            outputs_globs = []
            if desc:
                outputs_globs += glob.glob(FileOps.join_path(worker_path, "desc_*.json"))
            if weights_file:
                outputs_globs += glob.glob(FileOps.join_path(worker_path, "model_*.pth"))
            if performance:
                outputs_globs += glob.glob(FileOps.join_path(worker_path, "performance_*.json"))
            for _file in outputs_globs:
                FileOps.copy_file(_file, step_path)

    @classmethod
    def receive(cls, step_name, worker_id):
        """Get value from Shared Memory."""
        value = ShareMemory("{}.{}".format(step_name, worker_id)).get()
        if value:
            record = ReportRecord().from_dict(value)
        else:
            record = ReportRecord(step_name, worker_id)
        cls().add(record)
        return record

    @classmethod
    def broadcast(cls, record):
        """Broadcast one record to Shared Memory."""
        if not record:
            logging.warning("Broadcast Record is None.")
            return
        ShareMemory("{}.{}".format(record.step_name, record.worker_id)).put(record.serialize())
        cls().add(record)
        cls._save_worker_record(record.serialize())

    @classmethod
    def _save_worker_record(cls, record):
        step_name = record.get('step_name')
        worker_id = record.get('worker_id')
        _path = TaskOps().get_local_worker_path(step_name, worker_id)
        for record_name in ["desc", "performance"]:
            _file_name = None
            _file = None
            record_value = record.get(record_name)
            if not record_value:
                continue
            _file = None
            try:
                # for cars/darts save multi-desc
                if isinstance(record_value, list) and record_name == "desc":
                    for idx, value in enumerate(record_value):
                        _file_name = "desc_{}.json".format(idx)
                        _file = FileOps.join_path(_path, _file_name)
                        with open(_file, "w") as f:
                            json.dump(value, f)
                else:
                    _file_name = None
                    if record_name == "desc":
                        _file_name = "desc_{}.json".format(worker_id)
                    if record_name == "performance":
                        _file_name = "performance_{}.json".format(worker_id)
                    _file = FileOps.join_path(_path, _file_name)
                    with open(_file, "w") as f:
                        json.dump(record_value, f)
            except Exception as ex:
                logging.error("Failed to save {}, file={}, desc={}, msg={}".format(
                    record_name, _file, record_value, str(ex)))

    @classmethod
    def close(cls, step_name, worker_id):
        """Clear Shared Memory."""
        ShareMemory("{}.{}".format(step_name, worker_id)).close()

    def __repr__(self):
        """Override repr function."""
        return str(self.all_records)

    def update_report(self, worker_info):
        """Get finished worker's info, and use it to update target `generator`.

        Will get the finished worker's working dir, and then call the function
        `report.update(worker_info)`.
        :param worker_info: `worker_info` is the finished worker's info, usually
            a dict or list of dict include `step_name` and `worker_id`.
        :type worker_info: dict or list of dict.

        """
        if worker_info is None:
            return
        if not isinstance(worker_info, list):
            worker_info = [worker_info]
        for one_info in worker_info:
            step_name = one_info["step_name"]
            worker_id = one_info["worker_id"]
            logging.info("update report, step name: {}, worker id: {}".format(step_name, worker_id))
            try:
                result = ShareMemory("{}.{}".format(step_name, worker_id)).get()
                record = ReportRecord().from_dict(result)
                self.add(record)
                self.dump_report(step_name, record)
            except Exception:
                logging.error("Failed to upgrade report, step_name={}, worker_id={}.".format(step_name, worker_id))
                logging.error(traceback.format_exc())

    def _append_record_to_csv(self, record_name=None, step_name=None, record=None, mode='a'):
        """Transfer record to csv file."""
        local_output_path = os.path.join(TaskOps().local_output_path, step_name)
        logging.debug("recode to csv, local_output_path={}".format(local_output_path))
        if not record_name and os.path.exists(local_output_path):
            return
        file_path = os.path.join(local_output_path, "{}.csv".format(record_name))
        FileOps.make_base_dir(file_path)
        try:
            for key in record:
                if isinstance(record[key], dict) or isinstance(record[key], list):
                    record[key] = str(record[key])
            data = pd.DataFrame([record])
            if not os.path.exists(file_path):
                data.to_csv(file_path, index=False)
            elif os.path.exists(file_path) and os.path.getsize(file_path) and mode == 'a':
                data.to_csv(file_path, index=False, mode=mode, header=0)
            else:
                data.to_csv(file_path, index=False, mode=mode)
        except Exception as ex:
            logging.info('Can not transfer record to csv file Error: {}'.format(ex))

    def copy_pareto_output(self, step_name=None, worker_ids=[]):
        """Copy files related to pareto from  worker to output."""
        taskops = TaskOps()
        local_output_path = os.path.join(taskops.local_output_path, step_name)
        if not (step_name and os.path.exists(local_output_path)):
            return
        for worker_id in worker_ids:
            desDir = os.path.join(local_output_path, str(worker_id))
            FileOps.make_dir(desDir)
            local_worker_path = taskops.get_worker_subpath(step_name, str(worker_id))
            srcDir = FileOps.join_path(taskops.local_base_path, local_worker_path)
            copy_search_file(srcDir, desDir)

    @classmethod
    def load_records_from_model_folder(cls, model_folder):
        """Transfer json_file to records."""
        if not model_folder or not os.path.exists(model_folder):
            logging.error("Failed to load records from model folder, folder={}".format(model_folder))
            return []
        records = []
        pattern = FileOps.join_path(model_folder, "desc_*.json")
        files = glob.glob(pattern)
        for _file in files:
            try:
                with open(_file) as f:
                    worker_id = _file.split(".")[-2].split("_")[-1]
                    weights_file = os.path.join(os.path.dirname(_file), "model_{}.pth".format(worker_id))
                    if os.path.exists(weights_file):
                        sample = dict(worker_id=worker_id, desc=json.load(f), weights_file=weights_file)
                    else:
                        sample = dict(worker_id=worker_id, desc=json.load(f))
                    record = ReportRecord().load_dict(sample)
                    records.append(record)
            except Exception as ex:
                logging.info('Can not read records from json because {}'.format(ex))
        return records
