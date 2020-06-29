# -*- coding: utf-8 -*-
"""Random Algorithm used to Simple CNN."""
import logging
import copy
import random
from vega.search_space.search_algs import SearchAlgorithm
from vega.search_space.networks import NetworkDesc
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.codec import Codec
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SimpleRand(SearchAlgorithm):
    """Class of Random Search used to Simple CNN Example.

    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, search_space):
        super(SimpleRand, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space)
        args = self.cfg
        self.count = args.count
        self.max_num = args.max_num

    def search(self):
        """Search one NetworkDesc from search space.

        :return: search id, network desc
        :rtype: int, NetworkDesc
        """
        desc = copy.deepcopy(self.search_space.search_space)
        desc.pop('type')
        blocks = random.choice(desc.backbone.blocks)
        channels = random.choice(desc.backbone.channels)
        desc.backbone.blocks = blocks
        desc.backbone.channels = channels
        self.count += 1
        logging.info('Search No.{} sample'.format(self.count))
        return self.count, NetworkDesc(desc)

    def update(self, worker_path):
        """Update SimpleRand."""
        pass

    @property
    def is_completed(self):
        """Whether to complete algorithm."""
        return self.count >= self.max_num
