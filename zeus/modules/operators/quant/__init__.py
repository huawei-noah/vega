from zeus import is_tf_backend

if is_tf_backend():
    from .tensorflow_quant import *
else:
    from .pytorch_quant import *
