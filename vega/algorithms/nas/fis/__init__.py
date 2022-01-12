import vega
if vega.is_torch_backend():
    from .ctr_trainer_callback import CtrTrainerCallback
    from .autogroup_trainer_callback import AutoGroupTrainerCallback
    from .autogate_s1_trainer_callback import AutoGateS1TrainerCallback
    from .autogate_s2_trainer_callback import AutoGateS2TrainerCallback
    from .autogate_grda_s1_trainer_callback import AutoGateGrdaS1TrainerCallback
    from .autogate_grda_s2_trainer_callback import AutoGateGrdaS2TrainerCallback
