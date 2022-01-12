# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nodes for Modules."""


class Node(dict):
    """Node for Dag."""

    __slots__ = ['inputs', 'outputs', 'op_list', 'op_name']
    __support_ops__ = ['/Conv2D', '/FusedBatchNorm', '/MaxPool', '/MatMul', '/Mean', '/add', '/Pad',
                       '/Relu', 'Squeeze', '/Softmax']
    __support_ops_types__ = ['Mean']

    def __new__(cls, type_name=None, *args, **kwargs):
        """Create sub class according to type name."""
        for sub_class in cls.__subclasses__():
            if not sub_class.__class_type__ or not sub_class.__module_type__:
                raise Exception(f"__class_type__ and __module_type__ should be defined in class {sub_class}")
            sub_class_types = sub_class.__class_type__ if isinstance(sub_class.__class_type__, list) else [
                sub_class.__class_type__]
            if type_name in sub_class_types:
                return super(Node, cls).__new__(sub_class)
        return super(Node, cls).__new__(cls)

    def __init__(self, op_name=None, type_name=None, inputs=None, outputs=None, op_list=None):
        super(Node, self).__init__()
        self.type = type_name
        self.op_name = op_name
        self.inputs = inputs
        self.outputs = outputs
        self.op_list = op_list
        self.from_ops()
        # rename type into module type
        self.type = self.__module_type__

    def __setattr__(self, key, value):
        """Set value into dict."""
        self[key] = value

    def __getattr__(self, item):
        """Get value from dict."""
        return self.get(item)

    def from_ops(self):
        """Convert attrs from ops."""
        pass

    def __repr__(self):
        """Override repr."""
        return str(
            {k: v for k, v in self.items() if k not in self.__slots__ and not k.startswith('_') and v is not None})

    def to_json(self):
        """Convert items to dict."""
        res = {}
        for k, v in self.items():
            if k in self.__slots__ or k.startswith('_') or v is None:
                continue
            if isinstance(v, Node):
                res[k] = v.to_json()
            else:
                res[k] = v
        return res


class Sequential(Node):
    """Sequential for Node."""

    __class_type__ = 'Sequential'
    __module_type__ = 'Sequential'

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__(type_name=self.__class_type__, *args, **kwargs)
        self._idx = 0

    def append(self, value):
        """Append a item."""
        if value.name:
            name = value.name
        else:
            self._idx += 1
            name = str(self._idx)
        self.__setattr__(name, value)

    def to_json(self):
        """Convert item to dict."""
        res = super().to_json()
        modules = [k for k in self.keys() if
                   k not in self.__slots__ and not k.startswith('_') and k not in ['type', 'name']]
        if modules:
            res.update(dict(modules=modules))
        return res


class Add(Node):
    """Add Connections for Node."""

    __class_type__ = ['Add', 'AddV2']
    __module_type__ = 'Add'

    def __init__(self, *models, type_name=None, **kwargs):
        super(Add, self).__init__(type_name=type_name, **kwargs)
        for idx, model in enumerate(models):
            self.__setattr__(str(idx), model)


class Conv2DNode(Node):
    """Conv2D for Node."""

    __class_type__ = 'Conv2D'
    __module_type__ = 'Conv2d'

    def __init__(self, *args, **kwargs):
        self.kernel_size = None
        self.out_channels = None
        self.in_channels = None
        self.stride = None
        self.padding = None
        self.dilation = None
        self.bias = False
        self.bn = False
        super(Conv2DNode, self).__init__(*args, **kwargs)
        self.name = self.op_name.replace('/{}'.format(self.__class_type__), '') if self.op_name else ''

    def from_ops(self):
        """Convert attrs from ops."""
        import tensorflow as tf
        for op in self.op_list:
            if not isinstance(op, tf.Operation):
                continue
            if op.name.endswith('kernel') or op.name.endswith('weights'):
                self.kernel_size = op.outputs[0].shape.as_list()[0:2]
                self.out_channels = op.outputs[0].shape.as_list()[3]
            if op.name.endswith('bias'):
                self.bias = True
            if op.name.endswith("BiasAdd"):
                self.bn = True
            elif op.name.endswith(self.__class_type__):
                attr = op.node_def.attr
                data_format = str(attr.get('data_format').s, encoding='utf8')
                axis = 3 if 'NHWC' in data_format else 1
                in_channels = op.inputs[0]
                if in_channels.op.type == 'Pad':
                    pre_op_input = in_channels.op.inputs[0]
                    if pre_op_input.op.type == 'Transpose':
                        in_channels = pre_op_input.op.inputs[0]
                    else:
                        in_channels = in_channels.op.inputs[0]
                self.in_channels = in_channels.shape.as_list()[axis]
                attr = op.node_def.attr
                self.stride = list(attr.get('strides').list.i)[2]
                self.padding = str(attr.get('padding').s, encoding='utf8')
                self.dilation = list(attr.get('dilations').list.i)[1]


class BatchNorm2dNode(Node):
    """BatchNorm2D Node."""

    __class_type__ = ['FusedBatchNormV3', 'FusedBatchNorm']
    __module_type__ = 'BatchNorm2d'

    def __init__(self, *args, **kwargs):
        self.num_features = None
        super(BatchNorm2dNode, self).__init__(*args, **kwargs)
        for class_name in self.__class_type__:
            if self.op_name.endswith(class_name):
                self.name = self.op_name.replace('/{}'.format(class_name), '') if self.op_name else ''


class ReluNode(Node):
    """Relu Node."""

    __class_type__ = 'Relu'
    __module_type__ = 'Relu'

    def __init__(self, *args, **kwargs):
        super(ReluNode, self).__init__(*args, **kwargs)


class MaxPoolNode(Node):
    """MaxPool Node."""

    __class_type__ = 'MaxPool'
    __module_type__ = 'MaxPool2d'

    def __init__(self, *args, **kwargs):
        self.kernel_size = None
        self.stride = None
        self.padding = None
        super(MaxPoolNode, self).__init__(*args, **kwargs)
        self.name = self.op_name.replace('/{}'.format(self.__class_type__), '') if self.op_name else self.op_name

    def from_ops(self):
        """Convert attrs from ops."""
        op = self.op_list[0]
        import tensorflow as tf
        if not isinstance(op, tf.Operation):
            return
        attr = op.node_def.attr
        self.stride = list(attr.get('strides').list.i)[2]
        self.padding = str(attr.get('padding').s, encoding='utf8')
        self.kernel_size = list(attr.get('ksize').list.i)[3]


class LinearNode(Node):
    """Linear Node."""

    __class_type__ = ['MatMul', 'Softmax']
    __module_type__ = 'Linear'

    def __init__(self, *args, **kwargs):
        self.out_features = None
        self.in_features = None
        self.use_bias = None
        super(LinearNode, self).__init__(*args, **kwargs)
        op_name = self.op_name
        if op_name:
            for class_type in self.__class_type__:
                op_name = op_name.replace('/{}'.format(class_type), '')
        self.name = op_name

    def from_ops(self):
        """Convert attrs from ops."""
        import tensorflow as tf
        for op in self.op_list:
            if not isinstance(op, tf.Operation):
                continue
            if op.name.endswith('bias'):
                self.use_bias = True
            elif op.name.endswith('MatMul'):
                self.out_features = op.outputs[0].shape.as_list()[1]
                self.in_features = op.inputs[0].shape.as_list()[1]
            elif op.name.endswith('Softmax'):
                self.out_features = op.outputs[0].shape.as_list()[1]
                self.in_features = op.inputs[0].shape.as_list()[1]
        if self.inputs[0].op.type == 'Reshape':
            self.inputs = self.inputs[0].op.inputs


class MeanNode(Node):
    """Mean Node."""

    __class_type__ = 'Mean'
    __module_type__ = 'AdaptiveAvgPool2d'

    def __init__(self, *args, **kwargs):
        super(MeanNode, self).__init__(*args, **kwargs)


class SqueezeNode(Node):
    """Squeeze Node."""

    __class_type__ = 'Squeeze'
    __module_type__ = 'View'

    def __init__(self, *args, **kwargs):
        super(SqueezeNode, self).__init__(*args, **kwargs)


class PadNode(Node):
    """Padding Node."""

    __class_type__ = 'paddings'
    __module_type__ = 'Pad'

    def __init__(self, *args, **kwargs):
        self.kernel_size = None
        super(PadNode, self).__init__(*args, **kwargs)

    def from_ops(self):
        """Convert attrs from ops."""
        self.kernel_size = self.op_list[0].outputs[0].shape.as_list()
        pre_node_input = self.op_list[1].inputs[0]
        if pre_node_input.op.type == 'Identity':
            pre_node_input = pre_node_input.op.inputs[0]
        self.inputs = [pre_node_input]
