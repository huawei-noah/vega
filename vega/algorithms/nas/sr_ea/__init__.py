from .sr_random import SRRandom
from .sr_ea_codec import SRCodec
from .sr_mutate import SRMutate
import os

if os.environ['BACKEND_TYPE'] == 'PYTORCH':
    from .sr_ea_trainer_callback import SREATrainerCallback
