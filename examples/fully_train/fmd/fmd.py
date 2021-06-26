# -*- coding: utf-8 -*-.
"""Main function to run a FmdPipeStep."""
from vega.common import ClassFactory, ClassType
from networks.resnet_cifar import resnet_cifar


@ClassFactory.register(ClassType.NETWORK)
class FmdNetwork(resnet_cifar):
    """Wrapper a custom network resnet_cifar.

    :param desc: network description
    :type desc: dict
    """

    def __init__(self, **desc):
        depth = desc.get("depth")
        wide_factor = desc.get("wide_factor", 1)
        num_classes = desc.get("num_classes", 10)
        args = desc.get("args", None)
        super(FmdNetwork, self).__init__(depth, wide_factor, num_classes, args)
