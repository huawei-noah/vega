# -*- coding: utf-8 -*-
from .AutoContrast import AutoContrast
from .BboxTransform import BboxTransform
from .Brightness import Brightness
from .Color import Color
from .Compose import Compose
from .Compose_pair import Compose_pair
from .Contrast import Contrast
from .Cutout import Cutout
from .Equalize import Equalize
from .ImageTransform import ImageTransform
from .Invert import Invert
from .MaskTransform import MaskTransform
from .Posterize import Posterize
from .RandomCrop_pair import RandomCrop_pair
from .RandomHorizontalFlip_pair import RandomHorizontalFlip_pair
from .RandomMirrow_pair import RandomMirrow_pair
from .RandomRotate90_pair import RandomRotate90_pair
from .RandomVerticallFlip_pair import RandomVerticallFlip_pair
from .Rotate import Rotate
from .SegMapTransform import SegMapTransform
from .Sharpness import Sharpness
from .Shear_X import Shear_X
from .Shear_Y import Shear_Y
from .Solarize import Solarize
from .Translate_X import Translate_X
from .Translate_Y import Translate_Y
from .RandomColor_pair import RandomColor_pair
from .RandomGaussianBlur_pair import RandomGaussianBlur_pair
from .RandomRotate_pair import RandomRotate_pair
from .Rescale_pair import Rescale_pair
from .Normalize_pair import Normalize_pair

import zeus

if zeus.is_torch_backend():
    from .pytorch import *
elif zeus.is_tf_backend():
    from .tensorflow import *
elif zeus.is_ms_backend():
    pass
else:
    raise ValueError

try:
    from mmdet.datasets.extra_aug import PhotoMetricDistortion, Expand, ExtraAugmentation
except Exception:
    pass
