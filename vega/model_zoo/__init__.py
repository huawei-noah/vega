from .model_zoo import ModelZoo
from .torch_vision_model import import_all_torchvision_models


try:
    import_all_torchvision_models()
except Exception as e:
    import logging
    logging.warn("Failed to import torchvision models, msg={}".format(str(e)))
