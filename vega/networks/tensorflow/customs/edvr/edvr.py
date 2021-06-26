# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""EDVR network."""
import tensorflow as tf
from vega.common import ClassType, ClassFactory
from .edvr_util import PCDAlignment, TSAFusion, LAAlignment, SeparateNonLocal
from .arch_util import Conv2D, ActLayer, ConvModule, depth_to_space, resize, tf_split, ResBlockNoBN, ResBlockChnAtten


@ClassFactory.register(ClassType.NETWORK)
class EDVR(object):
    """EDVR network."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_frame=5, deformable_groups=8, num_extract_block=5,
                 num_reconstruct_block=10, center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True,
                 align_type='pcd', align_step=False, with_snl=False, with_chn_atten=False):
        self.with_tsa = with_tsa
        self.mid_channels = num_feat
        self.num_deform_groups = deformable_groups
        self.num_blocks_extraction = num_extract_block
        self.num_blocks_reconstruction = num_reconstruct_block
        self.center_frame_idx = center_frame_idx if center_frame_idx is not None else num_frame // 2
        self.num_frames = num_frame
        self.pcd_align = PCDAlignment(self.mid_channels, self.num_deform_groups)
        self.tsa_fusion = TSAFusion(self.mid_channels, self.num_frames, self.center_frame_idx)
        self.la_align = LAAlignment(num_feat=num_feat, radius=3, normalize=False)
        self.patial_align = self.pcd_align if align_type == 'pcd' else self.la_align
        self.align_step = align_step
        self.upsample_mode = 'bilinear'
        self.with_snl = with_snl
        self.separate_no_local = SeparateNonLocal(self.mid_channels)
        self.with_chn_atten = with_chn_atten

    def feature_extraction(self, x, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        """Feature extraction part of EDVR."""
        # extract LR features
        with tf.variable_scope('extraction'):
            # L1
            l1_feat = tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]])
            l1_feat = Conv2D(l1_feat, self.mid_channels, name='conv_first')
            l1_feat = ActLayer(act_cfg)(l1_feat)
            if self.with_chn_atten:
                l1_feat = ResBlockChnAtten(num_blocks=self.num_blocks_extraction,
                                           mid_channels=self.mid_channels)(l1_feat)
            else:
                l1_feat = ResBlockNoBN(num_blocks=self.num_blocks_extraction, mid_channels=self.mid_channels)(l1_feat)
            # l1_feat = ResBlockNoBN(num_blocks=self.num_blocks_extraction, mid_channels=self.mid_channels)(l1_feat)
            # L2
            l2_feat = ConvModule(l1_feat, self.mid_channels, strides=[2, 2], act_cfg=act_cfg, name='feat_l2_conv1')
            l2_feat = ConvModule(l2_feat, self.mid_channels, act_cfg=act_cfg, name='feat_l2_conv2')
            # L3
            l3_feat = ConvModule(l2_feat, self.mid_channels, strides=[2, 2], act_cfg=act_cfg, name='feat_l3_conv1')
            l3_feat = ConvModule(l3_feat, self.mid_channels, act_cfg=act_cfg, name='feat_l3_conv2')

            l1_feat = tf.reshape(l1_feat,
                                 [int(l1_feat.shape[0]) // self.num_frames, self.num_frames,
                                  int(l1_feat.shape[1]), int(l1_feat.shape[2]), -1])
            l2_feat = tf.reshape(l2_feat,
                                 [int(l2_feat.shape[0]) // self.num_frames, self.num_frames,
                                  int(l2_feat.shape[1]), int(l2_feat.shape[2]), -1])
            l3_feat = tf.reshape(l3_feat,
                                 [int(l3_feat.shape[0]) // self.num_frames, self.num_frames,
                                  int(l3_feat.shape[1]), int(l3_feat.shape[2]), -1])

            return l1_feat, l2_feat, l3_feat

    def reconstruction(self, feat, x_center, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        """Reconstruction part of EDVR."""
        # reconstruction
        with tf.variable_scope('reconstruction'):
            if self.with_chn_atten:
                out = ResBlockChnAtten(num_blocks=self.num_blocks_reconstruction, mid_channels=self.mid_channels)(feat)
            else:
                out = ResBlockNoBN(num_blocks=self.num_blocks_reconstruction, mid_channels=self.mid_channels)(feat)
            out = Conv2D(out, self.mid_channels * 2 ** 2, name='upsample1')
            out = depth_to_space(out, 2)
            out = Conv2D(out, self.mid_channels * 2 ** 2, name='upsample2')
            out = depth_to_space(out, 2)
            out = Conv2D(out, self.mid_channels, name='conv_hr')
            out = ActLayer(act_cfg)(out)
            out = Conv2D(out, 3, name='conv_last')

            base = resize(
                x_center, size=[x_center.shape[1] * 4, x_center.shape[2] * 4], align_corners=False,
                name='img_upsample', method=self.upsample_mode)
            base = tf.cast(base, tf.float32)
            out = tf.cast(out, tf.float32)
            out += base

            return out

    def __call__(self, x):
        """Forward function of EDVR."""
        # shape of x: [B,T_in,H,W,C]
        with tf.variable_scope('G'):
            x = tf.transpose(x, [0, 1, 3, 4, 2])
            x_list = tf.split(x, self.num_frames, axis=1)
            x_center = tf.squeeze(x_list[self.num_frames // 2], axis=1)

            # extract LR features
            l1_feat, l2_feat, l3_feat = self.feature_extraction(x)

            l1_feat_list = tf_split(l1_feat, self.num_frames, 1, keep_dims=False)
            l2_feat_list = tf_split(l2_feat, self.num_frames, 1, keep_dims=False)
            l3_feat_list = tf_split(l3_feat, self.num_frames, 1, keep_dims=False)

            ref_feats = [
                l1_feat_list[self.num_frames // 2],
                l2_feat_list[self.num_frames // 2],
                l3_feat_list[self.num_frames // 2]
            ]
            aligned_feat = []

            if self.align_step:
                act_cfg = dict(type='LeakyRelu', alpha=0.1)
                for i in range(self.num_frames):
                    neighbor_feats = [l1_feat_list[i], l2_feat_list[i], l3_feat_list[i]]
                    if i == 0 or i == self.num_frames - 1:
                        next = 1 if i == 0 else -1
                        temp_ref_feats = [l1_feat_list[i + next], l2_feat_list[i + next], l3_feat_list[i + next]]
                        l1_aligned_feat = self.patial_align(neighbor_feats, temp_ref_feats)
                        l2_aligned_feat = ConvModule(l1_aligned_feat, self.mid_channels, strides=[2, 2],
                                                     act_cfg=act_cfg, name='l2_aligned_{}'.format(i))
                        l3_aligned_feat = ConvModule(l2_aligned_feat, self.mid_channels, strides=[2, 2],
                                                     act_cfg=act_cfg, name='l3_aligned_{}'.format(i))
                        neighbor_feats = [l1_aligned_feat, l2_aligned_feat, l3_aligned_feat]
                    aligned_feat.append(self.patial_align(neighbor_feats, ref_feats))
            else:
                for i in range(self.num_frames):
                    neighbor_feats = [
                        l1_feat_list[i],
                        l2_feat_list[i],
                        l3_feat_list[i]
                    ]
                    aligned_feat.append(self.patial_align(neighbor_feats, ref_feats))

            aligned_feat = tf.stack(aligned_feat, axis=1)  # (n, t, h, w, c)

            if self.with_snl:
                aligned_feat = self.separate_no_local(aligned_feat)
            if self.with_tsa:
                feat = self.tsa_fusion(aligned_feat)
            else:
                aligned_feat = tf.transpose(aligned_feat, [0, 2, 3, 1, 4])
                aligned_feat = tf.reshape(aligned_feat,
                                          [aligned_feat.shape[0], aligned_feat.shape[1], aligned_feat.shape[2], -1])
                feat = Conv2D(aligned_feat, self.mid_channels, kernel_size=[1, 1], name='fusion')

            # reconstruction
            out = self.reconstruction(feat, x_center)
            out = tf.transpose(out, [0, 3, 1, 2])
            return out
