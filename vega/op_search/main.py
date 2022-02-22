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

"""This is the main script."""
import logging
import time
import os
from vega.common.arg_parser import argment_parser
from vega.common import FileOps
from vega import security
from vega.common.dag import DAG
from .utils import dag2compute, cal_error_threshold
from .op_generator import OpGenerator

logging.basicConfig(level=logging.INFO)

parser = argment_parser(desc='Automl-zero.')
parser.add_argument('--pop_num', type=int, default=1000, help='population number.')
parser.add_argument('--max_sample', type=int, default=2000000, help='max sample number.')
parser.add_argument('--save_csv', type=str, default='population.csv', help='the csv file to save the population.')
parser.add_argument('--input_data', type=str, default='./input.pkl', help='the input data file.')
parser.add_argument('--output_data', type=str, default='./out_mish.pkl', help='the real output data file.')
parser.add_argument('--threshold', type=int, default=1, help='the real output data file.')
parser.add_argument('--search_space', type=str, default=None, help='the search space yml file')

args = parser.parse_args()
security.check_args(args)


def main():
    """Process of vega op search."""
    nas = OpGenerator(population_num=args.pop_num, max_sample=args.max_sample, search_space=args.search_space)
    start_time = time.time()
    total_sample = 0
    valid_sample = 0
    invalid_sample = 0

    input_data = FileOps.load_pickle(args.input_data)
    real_output = FileOps.load_pickle(args.output_data)

    csv_file = args.save_csv
    if os.path.exists(csv_file):
        os.remove(csv_file)
    while not nas.is_finished():
        sample = nas.search()
        if sample is None:
            logging.debug("continue because sample is None.")
            continue
        logging.debug("Sample a formula: {}.".format(sample))
        dag = DAG()
        dag.from_dict(sample)
        res = dag2compute(dag, input_data)

        fitness = cal_error_threshold(res, real_output)
        if fitness <= args.threshold:
            if not nas._check_equivalence(fitness):
                valid_sample += 1
                # update population
                nas.update_fitness(num_id=total_sample, sample=sample, fitness=fitness)
                if fitness < 0.01:
                    logging.info("number: {} is a perfect sample, the fitness is: {}.".format(total_sample, fitness))
                elif fitness < 0.1:
                    logging.info("number: {} is a good sample, the fitness is: {}.".format(total_sample, fitness))
                else:
                    logging.info("number: {} is a valid sample, the fitness is: {}.".format(total_sample, fitness))
                logging.info("The sample is: {}.".format(sample))
            else:
                logging.info(
                    "number: {} is a equivalence, skip it, the fitness if {}.".format(total_sample, fitness))
        else:
            invalid_sample += 1
            logging.info(
                "number: {} is a invalid sample, because is not close enough, fitness is {}.".format(total_sample,
                                                                                                     fitness))
        total_sample += 1
        nas.population.to_csv(csv_file)

    end_time = time.time()
    logging.info(f"the best gene is: {nas.select_elite()}")
    logging.info(f"total time: {end_time - start_time}")
    logging.info(f"total samples: {total_sample}, valid: {valid_sample}, invalid: {invalid_sample}")
    logging.info(f"sample per seconds:{total_sample / (end_time - start_time)}")
    logging.info(f"valid sample per seconds:{valid_sample / (end_time - start_time)}")


if __name__ == "__main__":
    main()
