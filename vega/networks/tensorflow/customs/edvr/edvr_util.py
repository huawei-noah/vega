# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""EDVR util modules."""
import tensorflow as tf
from .arch_util import Conv2D, Conv3D, ActLayer, ConvModule, resize, tf_split
from .dcn import DCNPack


class PCDAlignment(object):
    """Module of yramid, cascading and deformable alignment."""

    def __init__(self, num_feat=64, deformable_groups=8, dcn_version='v2'):
        self.mid_channels = num_feat
        self.num_deform_groups = deformable_groups
        self.dcn_version = dcn_version
        self.num_groups = 1
        self.upsample_mode = 'bilinear'

    def __call__(self, neighbor_feats, ref_feats, act_cfg=dict(type='LeakyRelu', alpha=0.1),
                 name='pcd_align', reuse=tf.AUTO_REUSE):
        """Forward function of PCDAlignment."""
        with tf.variable_scope(name, reuse=reuse):
            # The number of pyramid levels is 3.
            if len(neighbor_feats) != 3 or len(ref_feats) != 3:
                raise Exception('The length of neighbor_feats and ref_feats must be both 3, '
                                'but got {} and {}'.format(len(neighbor_feats), len(ref_feats)))

            # Pyramids
            upsampled_offset, upsampled_feat = None, None
            for i in range(3, 0, -1):
                with tf.variable_scope('level{}'.format(i)):
                    offset = tf.concat([neighbor_feats[i - 1], ref_feats[i - 1]], axis=-1)
                    offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv1')
                    if i == 3:
                        offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv2')
                    else:
                        offset = tf.concat([offset, upsampled_offset], axis=-1)
                        offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv2')
                        offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv3')

                    feat = DCNPack(neighbor_feats[i - 1], offset, self.mid_channels, kernel_size=[3, 3], padding='same',
                                   num_deform_groups=self.num_deform_groups, num_groups=self.num_groups,
                                   name='dcn_l{}'.format(i), dcn_version=self.dcn_version)
                    if i == 3:
                        feat = ActLayer(act_cfg)(feat)
                    else:
                        feat = tf.concat([feat, upsampled_feat], axis=-1)
                        feat = ConvModule(feat, self.mid_channels, act_cfg=act_cfg if i == 2 else None,
                                          name='feat_conv')

                    if i > 1:
                        # upsample offset and features
                        # upsampled_offset = tf.image.resize_bilinear(
                        upsampled_offset = resize(
                            offset, size=[offset.shape[1] * 2, offset.shape[2] * 2], align_corners=False,
                            name='upsample_offset{}'.format(i), method=self.upsample_mode)
                        upsampled_offset = upsampled_offset * 2
                        # upsampled_feat = tf.image.resize_bilinear(
                        upsampled_feat = resize(
                            feat, size=[feat.shape[1] * 2, feat.shape[2] * 2], align_corners=False,
                            name='upsample_feat{}'.format(i), method=self.upsample_mode)

            # Cascading
            offset = tf.concat([feat, ref_feats[0]], axis=-1)
            offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='cas_offset_conv1')
            offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='cas_offset_conv2')
            feat = DCNPack(feat, offset, self.mid_channels, kernel_size=[3, 3], padding='same',
                           num_deform_groups=self.num_deform_groups, name='dcn_cas', dcn_version=self.dcn_version)
            feat = ActLayer(act_cfg)(feat)

            return feat


class TSAFusion(object):
    """Module of fusion with temporal and spatial attention."""

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        self.mid_channels = num_feat
        self.num_frames = num_frame
        self.center_frame_idx = center_frame_idx
        self.upsample_mode = 'bilinear'

    def __call__(self, aligned_feat, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        """Forward function of TSAFusion."""
        with tf.variable_scope('tsa_fusion'):
            n, t, h, w, c = list(map(int, aligned_feat.shape))
            # temporal attention
            aligned_feat_list = tf.split(aligned_feat, self.num_frames, axis=1)
            # aligned_feat_list = tf_split(aligned_feat, self.num_frames, axis=1)
            embedding_ref = Conv2D(
                tf.squeeze(aligned_feat_list[self.num_frames // 2], axis=1),
                # aligned_feat_list[self.num_frames // 2],
                self.mid_channels,
                name='temporal_attn1')
            emb = Conv2D(tf.reshape(aligned_feat, [-1, h, w, c]), self.mid_channels, name='temporal_attn2')
            emb = tf.reshape(emb, [n, t, h, w, -1])
            emb = tf.cast(emb, tf.float32)
            emb_list = tf_split(emb, self.num_frames, axis=1, keep_dims=False)

            corr_l = []  # correlation list
            for i in range(t):
                emb_neighbor = emb_list[i]
                corr = tf.reduce_sum(emb_neighbor * embedding_ref, axis=-1, keep_dims=True)  # (n, h, w, 1)
            #     corr_prob = tf.nn.sigmoid(corr)
            #     corr_l.append(corr_prob * aligned_feat_list[i])
            # aligned_feat = tf.concat(corr_l, axis=-1)
                corr_l.append(corr)
            corr_prob = tf.nn.sigmoid(tf.stack(corr_l, axis=1))  # (n, t, h, w, 1)
            aligned_feat = corr_prob * aligned_feat

            # fusion
            aligned_feat = tf.transpose(aligned_feat, [0, 2, 3, 1, 4])
            aligned_feat = tf.reshape(aligned_feat, [n, h, w, -1])
            feat = ConvModule(aligned_feat, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='feat_fusion')

            # spatial attention
            attn = ConvModule(aligned_feat, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg,
                              name='spatial_attn1')
            attn_max = tf.nn.max_pool2d(attn, 3, 2, 'SAME')
            attn_avg = tf.nn.avg_pool(attn, 3, 2, 'SAME')
            attn = ConvModule(tf.concat([attn_max, attn_avg], axis=-1), self.mid_channels, kernel_size=(1, 1),
                              act_cfg=act_cfg, name='spatial_attn2')
            # pyramid levels
            attn_level = ConvModule(attn, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg,
                                    name='spatial_attn_l1')
            attn_max = tf.nn.max_pool2d(attn_level, 3, 2, 'SAME')
            attn_avg = tf.nn.avg_pool(attn_level, 3, 2, 'SAME')
            attn_level = ConvModule(tf.concat([attn_max, attn_avg], axis=-1), self.mid_channels, act_cfg=act_cfg,
                                    name='spatial_attn_l2')
            attn_level = ConvModule(attn_level, self.mid_channels, act_cfg=act_cfg, name='spatial_attn_l3')
            # attn_level = tf.image.resize_bilinear(
            attn_level = resize(
                attn_level, size=[attn_level.shape[1] * 2, attn_level.shape[2] * 2], align_corners=False,
                name='upsample1', method=self.upsample_mode)

            attn = ConvModule(attn, self.mid_channels, act_cfg=act_cfg, name='spatial_attn3') + attn_level
            attn = ConvModule(attn, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn4')
            # attn = tf.image.resize_bilinear(
            attn = resize(
                attn, size=[attn.shape[1] * 2, attn.shape[2] * 2], align_corners=False,
                name='upsample2', method=self.upsample_mode)
            attn = Conv2D(attn, self.mid_channels, name='spatial_attn5')
            attn = ConvModule(attn, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn_add1')
            attn_add = Conv2D(attn, self.mid_channels, kernel_size=(1, 1), name='spatial_attn_add2')

            attn = tf.cast(attn, tf.float32)
            attn = tf.nn.sigmoid(attn)

            feat = tf.cast(feat, tf.float32)
            attn_add = tf.cast(attn_add, tf.float32)

            # after initialization, * 2 makes (attn * 2) to be close to 1.
            feat = feat * attn * 2 + attn_add
            return feat


class SeparateNonLocal(object):
    """Module of separated non-local."""

    def __init__(self, num_feat=64):
        self.num_feat = num_feat

    def __call__(self, x):
        """Forward function of separated no-local."""
        with tf.variable_scope('NonLocal'):
            B, T, H, W, C = x.get_shape().as_list()

            x1 = tf.cast(Conv3D(x, self.num_feat, name='spatial_1'), tf.float16)
            x2 = tf.cast(Conv3D(x, self.num_feat, name='spatial_2'), tf.float16)
            x3 = tf.cast(Conv3D(x, self.num_feat, name='spatial_3'), tf.float16)
            x1 = tf.reshape(tf.transpose(x1, [0, 2, 3, 1, 4]), [-1, H * W, T * C])
            x2 = tf.reshape(tf.transpose(x2, [0, 1, 4, 2, 3]), [-1, T * C, H * W])
            f = tf.nn.softmax(tf.matmul(x1, x2))  # B * (H*W) * (H*W)
            x3 = tf.reshape(tf.transpose(x3, [0, 2, 3, 1, 4]), [-1, H * W, T * C])
            y1 = tf.reshape(tf.matmul(f, x3), [-1, H, W, T, C])
            y1 = tf.cast(tf.transpose(y1, [0, 3, 1, 2, 4]), tf.float32)

            x1 = tf.cast(Conv3D(x, self.num_feat, name='channel_1'), tf.float16)
            x2 = tf.cast(Conv3D(x, self.num_feat, name='channel_2'), tf.float16)
            x3 = tf.cast(Conv3D(x, self.num_feat, name='channel_3'), tf.float16)
            x1 = tf.reshape(tf.transpose(x1, [0, 4, 2, 3, 1]), [-1, C, H * W * T])
            x2 = tf.reshape(x2, [-1, T * H * W, C])
            f = tf.nn.softmax(tf.matmul(x1, x2))  # B * C * C
            x3 = tf.reshape(tf.transpose(x3, [0, 4, 2, 3, 1]), [-1, C, H * W * T])
            y2 = tf.reshape(tf.matmul(f, x3), [-1, C, H, W, T])
            y2 = tf.cast(tf.transpose(y2, [0, 4, 2, 3, 1]), tf.float32)

            x1 = tf.cast(Conv3D(x, self.num_feat, name='temporal_1'), tf.float16)
            x2 = tf.cast(Conv3D(x, self.num_feat, name='temporal_2'), tf.float16)
            x3 = tf.cast(Conv3D(x, self.num_feat, name='temporal_3'), tf.float16)
            x1 = tf.reshape(x1, [-1, T, H * W * C])
            x2 = tf.reshape(tf.transpose(x2, [0, 2, 3, 4, 1]), [-1, H * W * C, T])
            f = tf.nn.softmax(tf.matmul(x1, x2))  # B * T * T
            x3 = tf.reshape(x3, [-1, T, H * W * C])
            y3 = tf.cast(tf.reshape(tf.matmul(f, x3), [-1, T, H, W, C]), tf.float32)

            return y1 + y2 + y3 + x


class LAAlignment(object):
    """Module of local aggregator."""

    def __init__(self, num_feat=64, radius=3, normalize=False):
        self.num_feat = num_feat
        self.upsample_mode = 'bilinear'
        self.local_agg = LocalAggregator(radius=radius, nf=num_feat, normalize=normalize)

    def __call__(self, neighbor_feats, ref_feats):
        """Forward function of local aggregator alignment."""
        with tf.variable_scope('LA_Alignment', reuse=tf.AUTO_REUSE):
            aligned_feats = []
            for i in range(3, 0, -1):
                neighbor_feat = neighbor_feats[i - 1]
                ref_feat = ref_feats[i - 1]
                aligned_feat = self.local_agg(ref_feat, neighbor_feat, name='local_agg_{}'.format(i))
                while i > 1:
                    aligned_feat = resize(aligned_feat, size=[aligned_feat.shape[1] * 2, aligned_feat.shape[2] * 2],
                                          align_corners=False, method=self.upsample_mode)
                    i -= 1
                aligned_feats.append(aligned_feat)
            feat = tf.concat(aligned_feats, axis=-1)
            feat = ConvModule(feat, self.num_feat, act_cfg=dict(type='LeakyRelu', alpha=0.1), name='la')
            return feat


class LocalAggregator(object):
    """Local Aggregator."""

    def __init__(self, radius, nf, normalize=False):
        self.normalize = normalize
        self.radius = radius
        self.num_feat = nf
        self.offsets = [(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]
        self.act_cfg = dict(type='LeakyRelu', alpha=0.1)

    def __call__(self, ref_feat, feat, name='local_agg'):
        """Forward function of local aggregator."""
        with tf.variable_scope(name):
            B, H, W, C = list(map(int, ref_feat.shape))
            ref_feat = ConvModule(ref_feat, self.num_feat // 2, act_cfg=self.act_cfg, name='conv_ref')
            pad_feat = tf.keras.layers.ZeroPadding2D(padding=self.radius)(feat)
            feat = ConvModule(pad_feat, self.num_feat // 2, act_cfg=self.act_cfg, name='conv_feat')
            if self.normalize:
                ref_feat = tf.nn.l2_normalize(ref_feat, dim=-1)
                feat = tf.nn.l2_normalize(feat, dim=-1)
            correlations = []
            shiftedFeats = []
            for i, j in self.offsets:
                shiftedFeat = tf.image.crop_to_bounding_box(pad_feat, self.radius + i, self.radius + j, H, W)
                shiftedFeats.append(shiftedFeat)

                shiftedFeat1 = tf.image.crop_to_bounding_box(feat, self.radius + i, self.radius + j, H, W)
                correlation = tf.reduce_sum(tf.math.multiply(ref_feat, shiftedFeat1), axis=-1, keepdims=False)
                correlations.append(correlation)
            correlations = tf.stack(correlations, axis=1)
            correlations = tf.reshape(correlations, (-1, len(self.offsets)))
            correlations = tf.nn.softmax(correlations, axis=1)
            correlations = tf.reshape(correlations, (B, -1, H, W))
            indices = [i for i in range(B) for _ in range(C)]
            correlations = tf.gather(correlations, indices)
            shiftedFeats = tf.stack(shiftedFeats, axis=1)
            shiftedFeats = tf.reshape(shiftedFeats, (-1, len(self.offsets), H, W))
            result = tf.reduce_sum(tf.math.multiply(shiftedFeats, correlations), axis=1, keepdims=False)
            result = tf.reshape(result, (B, H, W, C))
            return result
