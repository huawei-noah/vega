# -*- coding: utf-8 -*-
"""Random Algorithm used to Simple CNN."""
import logging
import copy
import random
from vega.search_space.search_algs import SearchAlgorithm
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.codec import Codec
from vega.core.common.file_ops import FileOps


class SimpleRandConfig(object):
    """BackboneNas Config."""

    count = 0
    max_num = 50


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SimpleRand(SearchAlgorithm):
    """Class of Random Search used to Simple CNN Example.

    :param search_space: search space
    :type search_space: SearchSpace
    """

    config = SimpleRandConfig

    def __init__(self, search_space):
        super(SimpleRand, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space)
        self.count = self.config.count
        self.max_num = self.config.max_num

    def search(self):
        """Search one NetworkDesc from search space.

        :return: search id, network desc
        :rtype: int, NetworkDesc
        """
        desc = copy.deepcopy(self.search_space)
        desc.pop('type')
        blocks = random.choice(desc.backbone.blocks)
        channels = random.choice(desc.backbone.channels)
        desc.backbone.blocks = blocks
        desc.backbone.channels = channels
        self.count += 1
        logging.info('Search No.{} sample'.format(self.count))
        return self.count, desc

    def update(self, worker_path):
        """Update SimpleRand."""
        pass

    @property
    def is_completed(self):
        """Whether to complete algorithm."""
        return self.count >= self.max_num
