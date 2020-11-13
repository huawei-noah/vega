import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')

from . import core  # noqa: E402
from .core import run, init_local_cluster_args, set_backend, module_existed  # noqa: E402
from .core import is_gpu_device, is_npu_device, is_torch_backend, is_tf_backend, is_ms_backend
