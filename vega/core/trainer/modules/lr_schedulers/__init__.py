from .lr_scheduler import LrScheduler
import os

if os.environ['BACKEND_TYPE'] == 'PYTORCH':
    from .warmup_scheduler import WarmupScheduler
elif os.environ['BACKEND_TYPE'] == 'TENSORFLOW':
    from .multistep_warmup import MultiStepLrWarmUp
    from .cosine_annealing import CosineAnnealingLR
