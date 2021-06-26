__version__ = "1.5.0"


import sys
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')


from .common.backend_register import *
from .common.class_factory import ClassFactory, ClassType
from .core import run, init_cluster_args, module_existed
from .trainer.trial_agent import TrialAgent


def network(name, **kwargs):
    """Return network."""
    return ClassFactory.get_cls(ClassType.NETWORK, name)(**kwargs)


def dataset(name, **kwargs):
    """Return dataset."""
    return ClassFactory.get_cls(ClassType.DATASET, name)(**kwargs)


def trainer(name="Trainer", **kwargs):
    """Return trainer."""
    return ClassFactory.get_cls(ClassType.TRAINER, name)(**kwargs)
