# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Define BackboneNasCodec."""

from .utils.str2dict import str2dict
from .utils.resnet_variant_det_codec import ResNetVariantDetCodec
from .utils.resnext_variant_det_codec import ResNeXtVariantDetCodec
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class AutoLaneNasCodec(Codec):
    """AutoLaneNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace
    """

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNasCodec."""
        super(AutoLaneNasCodec, self).__init__(search_space=search_space, **kwargs)

    def encode(self, sample_desc, is_random=False):
        """Encode backbone.

        :param sample_desc: a sample desc to encode.
        :type sample_desc: dict
        :param is_random: if use random to encode, default is False.
        :type is_random: bool
        :return: an encoded sample.
        :rtype: dict
        """
        backbone_name, backbone_params = list(sample_desc['backbone'].items()).pop()
        backbone_optional_list = {'ResNetVariantDet', 'ResNeXtVariantDet'}
        if backbone_name not in backbone_optional_list:
            raise NotImplementedError(f'Only {backbone_optional_list} is support in auto_lane algorithm')
        CodecSpec = globals()[f'{backbone_name}Codec']
        backbone_code = CodecSpec(**CodecSpec.random_sample(base_channel=backbone_params['base_channel'],
                                                            base_depth=backbone_params['base_depth'])).arch_code
        ffm_name, ffm_params = list(sample_desc['neck'].items()).pop()
        ffm_optional_list = ['FeatureFusionModule']
        if ffm_name not in ffm_optional_list:
            raise NotImplementedError(f'Only {ffm_optional_list} is support in auto_lane algorithm')
        ffm_code = ffm_params['arch_code']
        return f'{backbone_code}+{ffm_code}'

    def decode(self, sample):
        """Decode backbone to description.

        :param sample: input sample to decode.
        :type sample: dict
        :return: return a decoded sample desc.
        :rtype: dict
        """
        if 'code' not in sample:
            raise ValueError('No code to decode in sample:{}'.format(sample))
        backbone_code, ffm_code = sample['code'].split('+')

        decoder_map = dict(x=ResNeXtVariantDetCodec, r=ResNetVariantDetCodec)
        CodecSpec = decoder_map.get(backbone_code[0], None)
        if CodecSpec is None:
            raise NotImplementedError(f'Only {decoder_map} is support in auto_lane algorithm')
        generator = CodecSpec(**CodecSpec.arch_decoder(backbone_code))
        backbone_desc = str2dict(generator.config)
        neck_desc = dict(
            arch_code=ffm_code,
            name='FeatureFusionModule',
            in_channels=backbone_desc['out_channels'],
        )
        head_desc = dict(
            base_channel=128 + 128 + backbone_desc['out_channels'][2] if ffm_code != '-' else
            backbone_desc['out_channels'][2],
            num_classes=2,
            up_points=73,
            down_points=72,
            name='AutoLaneHead'
        )
        detector = dict(
            name='AutoLaneDetector',
            modules=['backbone', 'neck', 'head'],
            num_class=2,
            method=sample['method'],
            code=sample['code'],
            backbone=backbone_desc,
            neck=neck_desc,
            head=head_desc
        )
        return dict(modules=['detector'], detector=detector)
