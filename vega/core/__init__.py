from .run import run, env_args, init_local_cluster_args
from .backend_register import set_backend
from zeus import is_gpu_device, is_npu_device, is_torch_backend, is_tf_backend, is_ms_backend
from zeus.trainer import *
# from .evaluator import *
from zeus.common import FileOps, TaskOps, UserConfig, module_existed
