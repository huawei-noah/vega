#!/usr/bin/env python
"""Display reward and loss infomation into tensorboard."""
import os

from vega.visual.tensorboarder import clean_board_dir
from vega.visual.tensorboarder import SummaryBoard
from vega.common.util.default_xt import XtBenchmarkConf as bm_conf
from vega.common.util.get_xt_config import parse_xt_multi_case_paras
from vega.common.util.evaluate_xt import parse_benchmark_args
from vega.common.util.evaluate_xt import read_train_records, read_train_event_id


def display_rewards(args, stage):
    """Create utils for display, support multi & single config file."""
    for _conf_file in args.config_file:
        if not os.path.isfile(_conf_file):
            print("config file: '{}' invalid, continue!".format(_conf_file))
            continue

        print("processing config file: '{}' ".format(_conf_file))
        multi_case_paras = parse_xt_multi_case_paras(_conf_file)

        for _once_paras in multi_case_paras:
            handle_once_local_data_record(_once_paras, args.use_index, stage)


def parse_xt_train_config(yaml_obj):
    """Create utils for parse xt config file."""
    env = yaml_obj.get("env_para")
    alg = yaml_obj.get("alg_para")
    _model = yaml_obj.get("model_para")
    alg["model_info"] = _model
    agent = yaml_obj.get("agent_para")

    return env, alg, agent


def handle_once_local_data_record(case_paras, use_index, stage="eval",
                                  clear_tensorboard=True):
    """Handle the record from local file."""
    env_info, alg_info, agent_info = parse_xt_train_config(case_paras)
    # NOTE: model info will insert into alg_info,  as "model_info"
    benchmark_info = case_paras.get("benchmark", dict())

    bm_args = parse_benchmark_args(env_info, alg_info, agent_info, benchmark_info)

    records = read_train_records(bm_args, use_index, stage)

    prefix_display_name = "_".join([env_info["env_name"], env_info["env_info"]["name"]])
    _train_event_id = read_train_event_id(bm_args)
    case_tb_dir = "_".join(
        [prefix_display_name, alg_info["alg_name"], str(_train_event_id)]
    )
    print("case_tb_dir: ", case_tb_dir)
    if clear_tensorboard:
        clean_board_dir(os.path.join(bm_conf.default_tb_path, case_tb_dir))

    write2board(stage, records, use_index, case_tb_dir)


def write2board(stage, record_dict, use_index, case_tb_dir):
    """Write record into tensorboard, include, loss, reward etc."""
    if use_index == "step":
        x_key = "sample_step"
    elif use_index == "sec":
        x_key = "elapsed_sec"
    else:
        raise KeyError("need in 'step' or 'sec', get: {}".format(use_index))

    if stage == "eval":
        display_list = ["eval_reward"]
    elif stage == "both":
        display_list = ["eval_reward", "train_reward"]
    else:
        raise KeyError("invalid stage para-{}".format(stage))

    summary = SummaryBoard(archive_root=bm_conf.default_tb_path, fixed_path=case_tb_dir)
    for name in display_list:
        for x_val, value in zip(record_dict[x_key], record_dict[name]):
            summary.add_scalar(name, value, x_val)

    del summary
