# -*- coding:utf-8 -*-
"""Run benchmark."""
import os
import sys
import vega


def _append_env():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, dir_path)
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = dir_path
    else:
        os.environ["PYTHONPATH"] += ":{}".format(dir_path)


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3] and not sys.argv[1].endswith(".yml"):
        print("Usage:")
        print("    python3 ./run_benchmark.py <algorithm's config file> [pytorch(default)]|[p]|[tensorflow]|[t]")
        print("for example:")
        print("    python3 ./run_benchmark.py ./nas/cars/cars.yml")
        print("    python3 ./run_benchmark.py ./nas/cars/cars.yml pytorch")
        print("    python3 ./run_benchmark.py ./nas/cars/cars.yml tensorflow")
    # set backend
    if len(sys.argv) == 3 and sys.argv[2] in ["pytorch", "p", "tensorflow", "t"]:
        if sys.argv[2] in ["pytorch", "p"]:
            vega.set_backend("pytorch")
        else:
            vega.set_backend("tensorflow")
    # import class lazily
    cfg_file = sys.argv[1]
    if cfg_file.endswith("fmd.yml"):
        _append_env()
        from algs.fully_train.fmd import FmdNetwork
    elif cfg_file.endswith("spnas.yml"):
        _append_env()
        import vega.algorithms.nas.sp_nas
    # run vega
    vega.run(sys.argv[1])
