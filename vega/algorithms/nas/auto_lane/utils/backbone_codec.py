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

"""The base class of backbone codec."""


class Backbone():
    """This is the class of Backbone.

    :param with_neck: if the model with neck
    :type with_neck: bool
    :param train_from_scratch: if the model need train from scratch
    :type train_from_scratch: bool
    :param fore_part: the fore part of the model
    :type fore_part: Module
    """

    type = 'Backbone'
    component = 'backbone'
    quest_dict = dict(train_from_scratch=('optimizer', 'train_from_scratch'),
                      with_neck=('detector', 'with_neck'))

    def __init__(self, with_neck=True, train_from_scratch=True, fore_part=None, *args, **kwargs):
        super(Backbone, self).__init__()
        """Construct base class."""
        # set quested params
        self.train_from_scratch = train_from_scratch
        self.with_neck = with_neck
        # set other params
        if self.train_from_scratch:
            self.norm_cfg = dict(type='BN', requires_grad=True)
            self.conv_cfg = dict(type='Conv')
        else:
            if self.with_neck:
                self.norm_cfg = dict(type='BN', requires_grad=True)
            else:
                self.norm_cfg = dict(type='BN', requires_grad=False)
            self.conv_cfg = None

    @property
    def checkpoint_name(self):
        """Return the ckeckpoint name.

        :return: checkpoint name (str)
        :rtype: str
        """
        return self.name

    @property
    def pretrained(self):
        """Return the pretrain model.

        :return: the pretrain model path (str)
        :rtype: str
        """
        if self.train_from_scratch:
            return None
        else:
            raise NotImplementedError

    @property
    def input_size(self):
        """Return input_size.

        :return: input_size (tuple)
        :rtype: tuple
        """
        return self.quest_from(self.model['dataset'], 'img_scale')['test']

    def get_model(self):
        """Return model info.

        :return: model info
        :rtype: correspond model type
        """
        raise NotImplementedError

    @property
    def size_info(self):
        """Return model size info.

        :return: size info
        :rtype: str
        """
        pass

    @property
    def flops_ratio(self):
        """Return flops ratio.

        :return: flops ratio (int)
        :rtype: int
        """
        return 1

    @property
    def flops(self):
        """Return flops.

        :return: flops (int)
        :rtype: int
        """
        return self.size_info['FLOPs']

    @property
    def mac(self):
        """Return model mac.

        :return: mac (str)
        :rtype: str
        """
        return self.size_info['MAC']

    @property
    def params(self):
        """Return params.

        :return: params
        :rtype: dict
        """
        return self.size_info['params']

    @staticmethod
    def arch_decoder(arch: str):
        """Arch decoder."""
        pass

    @classmethod
    def set_from_arch_string(cls, arch_string, fore_part=None, **kwargs):
        """Return model params.

        :param arch_string: model arch
        :type arch_string: str
        :param fore_part: fore part model
        :type fore_part: dict
        :return: params (dict)
        """
        params = dict(fore_part=fore_part)
        params.update(cls.arch_decoder(arch_string))
        params.update(cls.quest_param(fore_part=fore_part, **kwargs))
        params.update(kwargs)
        return cls(**params)
