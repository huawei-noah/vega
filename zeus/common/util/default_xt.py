"""Make default configure for benchmark function."""


class XtBenchmarkConf(object):
    """Make benchmark conf, user also can re-set it."""

    default_db_root = "/tmp/.xt_data/sqlite"  # could set path by yourself
    default_id = "xt_default_benchmark"
    defalut_log_path = "/tmp/.xt_data/logs"
    default_tb_path = "/tmp/.xt_data/tensorboard"
    default_plot_path = "/tmp/.xt_data/plot"
    default_train_interval_per_eval = 200
