import vega
from .lr_scheduler import LrScheduler

if vega.is_torch_backend():
    from .warmup_scheduler_torch import WarmupScheduler
    from .scheduler_dict import SchedulerDict
elif vega.is_tf_backend():
    from .warmup_scheduler_tf import WarmupScheduler
    from .multistep import MultiStepLR
    from .cosine_annealing import CosineAnnealingLR
    from .step_lr import StepLR
    from .ca_restart_tf import CosineAnnealingRestartLR
elif vega.is_ms_backend():
    from .ms_lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR, PolyLR, WarmupScheduler
