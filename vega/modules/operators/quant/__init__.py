import vega


if vega.is_tf_backend():
    from .tensorflow_quant import *
elif vega.is_torch_backend():
    from .pytorch_quant import *
