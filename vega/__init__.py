import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')

from . import config  # noqa: E402
from . import core    # noqa: E402
from .core import run, init_local_cluster_args, set_zone, set_backend, module_existed  # noqa: E402
# from . import algorithms   # noqa: E402
# from . import model_zoo    # noqa: E402
