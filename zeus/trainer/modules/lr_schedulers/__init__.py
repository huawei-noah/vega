from .lr_scheduler import LrScheduler
import zeus

if zeus.is_torch_backend():
    from .warmup_scheduler import WarmupScheduler
elif zeus.is_tf_backend():
    from .multistep_warmup import MultiStepLRWarmUp
    from .cosine_annealing import CosineAnnealingLR
    from .step_lr import StepLR
