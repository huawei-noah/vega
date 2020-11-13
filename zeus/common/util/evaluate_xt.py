"""Make setup configs for xt server."""
import os
import csv
from datetime import datetime
from copy import deepcopy
from zeus.common.util.hw_cloud_helper import XT_HWC_WORKSPACE

TRAIN_CONFIG_YAML = "train_config.yaml"
TRAIN_RECORD_CSV = "records.csv"
DEFAULT_ARCHIVE_DIR = "xt_archive"
DEFAULT_FIELDS = [
    "train_index",
    "elapsed_sec",
    "sample_step",
    "train_reward",
    "eval_reward",
    "eval_criteria",
    "loss",
    "eval_name",
    "agent_id",
]

__all__ = [
    "parse_benchmark_args",
    "make_workspace_if_not_exist",
    "read_train_records",
    "find_train_info",
    "fetch_train_event",
    "read_train_event_id",
    "get_train_model_path_from_config",
    "read_train_records_from_config",
    "TRAIN_CONFIG_YAML",
    "TRAIN_RECORD_CSV",
    "DEFAULT_FIELDS",
    "DEFAULT_ARCHIVE_DIR",
    "get_bm_args_from_config"
]


def parse_benchmark_args(env_para, alg_para, agent_para, benchmark_info):
    """
    Parse benchmark information, simple the api for learner.

    Args:
    ----
        env_para:
        alg_para:
        agent_para:
        benchmark_info:
    """
    if not benchmark_info:
        benchmark_info = dict()
    bm_info_dict = {
        "env": deepcopy(env_para),
        "alg": deepcopy(alg_para),
        "agent": deepcopy(agent_para),
        "archive_root": deepcopy(benchmark_info.get("archive_root")),
        "bm_id": deepcopy(benchmark_info.get("id")),
        "bm_board": deepcopy(benchmark_info.get("board")),
        "bm_eval": deepcopy(benchmark_info.get("eval", {})),
    }
    return bm_info_dict


def make_dirs_if_not_exist(path):
    if path.startswith("s3://"):
        import moxing as mox

        if mox.file.is_directory(path) is False:
            mox.file.make_dirs(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def get_default_archive_path():
    """
    Makeup default archive path.

    Unify the archive path between local machine and cloud.
    """
    if not XT_HWC_WORKSPACE:
        return os.path.join(os.path.expanduser("~"), DEFAULT_ARCHIVE_DIR)
    else:
        return os.path.join(XT_HWC_WORKSPACE, DEFAULT_ARCHIVE_DIR)


def get_default_benchmark_id(benchmark_args):
    _env_name = benchmark_args.get("env", dict()).get("env_info", dict()).get("name")
    _alg_name = benchmark_args.get("alg", dict()).get("alg_name")
    return "_".join(["xt", _env_name, _alg_name])


def add_timestamp_postfix(str_base, connector):
    return "{}".format(connector).join(
        [str_base, datetime.now().strftime("%Y%m%d%H%M%S")]
    )


def __get_archive_bm_basic_info(benchmark_args):
    archive_root = benchmark_args.get("archive_root")
    if not archive_root:
        archive_root = get_default_archive_path()

    bm_id = benchmark_args["bm_id"]
    if not bm_id:
        bm_id = get_default_benchmark_id(benchmark_args)

    if not os.path.exists(archive_root):
        os.makedirs(archive_root)

    return os.path.abspath(archive_root), bm_id


def __make_workspace(benchmark_args, connector="+"):
    """
    Make workspace path join with connector.

    Support user's fix path within connector character.
    """
    archive_root, bm_id = __get_archive_bm_basic_info(benchmark_args)
    if connector not in bm_id:
        bm_id = add_timestamp_postfix(bm_id, connector)

    return os.path.join(archive_root, bm_id), archive_root, bm_id


def make_workspace_if_not_exist(benchmark_args, subdir="models"):
    """Make workspace if not exist."""
    workspace, archive_root, bm_id = __make_workspace(benchmark_args)
    make_dirs_if_not_exist(workspace)
    if isinstance(subdir, str):
        make_dirs_if_not_exist(os.path.join(workspace, subdir))
    elif isinstance(subdir, list):
        for path in subdir:
            make_dirs_if_not_exist(os.path.join(workspace, path))

    return workspace, archive_root, bm_id


def fetch_train_event(archive_root, bm_id, single=False):
    """
    Combine once train event path with the archive path, id and timestamp.

    order: special > newest
    :param archive_root:
    :param bm_id:
    :param single: if return single id
    :return:
    """
    event_path = os.path.join(archive_root, bm_id)
    if os.path.exists(os.path.join(event_path, TRAIN_RECORD_CSV)):
        return event_path

    event_list = list()
    event_list.extend(
        [_event for _event in os.listdir(archive_root) if _event.startswith(bm_id)]
    )
    event_list.sort(reverse=True)
    for _event in event_list:
        if os.path.exists(os.path.join(archive_root, _event, TRAIN_RECORD_CSV)):
            if single:
                return _event
            else:
                return os.path.join(archive_root, _event)
    if "+" in event_path:
        return event_path
    # raise ValueError("miss match under: {}".format(archive_root))


def find_train_info(train_event_path, use_index, stage):
    """Find train info."""
    with open(os.path.join(train_event_path, TRAIN_RECORD_CSV), "r") as rf:
        dict_reader = csv.DictReader(rf)
        record_data = [_d for _d in dict_reader]

    ret_dict = dict()

    def _fetch_field_val(key):
        # _field_index = get_field_index(key)
        return {key: [_row[key] for _row in record_data]}

    if use_index == "step":
        ret_dict.update(_fetch_field_val("sample_step"))
    elif use_index == "sec":
        ret_dict.update(_fetch_field_val("elapsed_sec"))
    else:
        raise KeyError("non-support index-{}".format(use_index))

    if stage == "eval":
        reward_key_list = [
            "eval_reward",
        ]
    elif stage == "both":
        reward_key_list = ["eval_reward", "train_reward"]
    elif stage == "all":
        reward_key_list = DEFAULT_FIELDS
    else:
        raise KeyError("stage para invalid, got: {}".format(stage))

    for _field in reward_key_list:
        ret_dict.update(_fetch_field_val(_field))

    return ret_dict


def read_train_event_id(benchmark_args):
    """Read train event id."""
    archive_root, bm_id = __get_archive_bm_basic_info(benchmark_args)

    return fetch_train_event(archive_root, bm_id, single=True)


def __get_wp_from_bm_args(bm_args):
    archive_root, bm_id = __get_archive_bm_basic_info(bm_args)
    if not os.path.exists(archive_root):
        os.makedirs(archive_root)

    workspace = fetch_train_event(archive_root, bm_id)
    return workspace


def read_train_records(benchmark_args, use_index="step", stage="both"):
    """Read train records."""
    workspace = __get_wp_from_bm_args(benchmark_args)
    return find_train_info(workspace, use_index, stage)


def get_bm_args_from_config(config):
    """Get bm args from config."""
    alg_para = config["alg_para"]
    env_para = config["env_para"]
    agent_para = config["agent_para"]
    model_info = config["model_para"]
    alg_para["model_info"] = model_info
    bm_info = config.get("benchmark", dict())
    return parse_benchmark_args(env_para, alg_para, agent_para, bm_info)


def read_train_records_from_config(config, use_index="step", stage="both"):
    """Read train records from config."""
    bm_args = get_bm_args_from_config(config)
    return read_train_records(bm_args, use_index, stage)


def get_train_model_path_from_config(config):
    """Get train model path from config."""
    bm_args = get_bm_args_from_config(config)
    workspace = __get_wp_from_bm_args(bm_args)
    return os.path.join(workspace, "models")
