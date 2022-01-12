import vega


if vega.is_tf_backend():
    from .tensorflow_quant import QuantConv, quant_custom_ops
elif vega.is_torch_backend():
    from .pytorch_quant import Quantizer, QuantConv, quant_custom_ops
