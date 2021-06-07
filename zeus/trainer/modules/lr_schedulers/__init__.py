from .lr_scheduler import LrScheduler
import zeus

if zeus.is_torch_backend():
    from .warmup_scheduler_torch import WarmupScheduler
    from .scheduler_dict import SchedulerDict
elif zeus.is_tf_backend():
    from .warmup_scheduler_tf import WarmupScheduler
    from .multistep import MultiStepLR
    from .cosine_annealing import CosineAnnealingLR
    from .step_lr import StepLR
    from .ca_restart_tf import CosineAnnealingRestartLR
elif zeus.is_ms_backend():
    from .ms_lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
