import zeus


if zeus.is_tf_backend():
    from .tensorflow_quant import *
elif zeus.is_torch_backend():
    from .pytorch_quant import *
