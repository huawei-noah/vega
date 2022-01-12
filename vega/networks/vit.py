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

"""Implementation of Visison Transformer(ViT)."""
import logging
from functools import partial
from collections import OrderedDict
import numpy as np
from vega.modules.operators import ops
from vega.common.class_factory import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.connections import Sequential

_logger = logging.getLogger(__name__)


class Mlp(Module):
    """Mlp layer in Transformer."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ops.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ops.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = ops.Linear(hidden_features, out_features)
        self.drop = ops.Dropout(drop)

    def call(self, x):
        """Forward mlp layer."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(Module):
    """Attention layer in Transformer."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = ops.Linear(dim, dim * 3, qkv_bias)
        self.attn_drop = ops.Dropout(attn_drop)
        self.proj = ops.Linear(dim, dim)
        self.proj_drop = ops.Dropout(proj_drop)

    def call(self, x):
        """Forward Attention layer."""
        B, N, C = x.shape
        qkv = ops.View((B, N, 3, self.num_heads, C // self.num_heads))(self.qkv(x))
        qkv = ops.Permute((2, 0, 3, 1, 4))(qkv)
        q = qkv[0:1]
        k = qkv[1:2]
        v = qkv[2:3]
        q = ops.Squeeze(0)(q)
        k = ops.Squeeze(0)(k)
        v = ops.Squeeze(0)(v)
        attn = ops.matmul(q, ops.Transpose(2, 3)(k)) * self.scale
        attn = ops.softmax(attn, -1)
        attn = self.attn_drop(attn)
        x = ops.Transpose(1, 2)(ops.matmul(attn, v))
        x = ops.View((B, N, C))(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(Module):
    """Block of Transformer, which contains one Attenson layer and one MLP layaer."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=ops.gelu, norm_layer=ops.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ops.DropPath(drop_path) if drop_path > 0. else ops.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def call(self, x):
        """Forward block."""
        x = x + self.drop_path(self.attn(self.norm1(x)))  # x shape is (1, 577, 768)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # x shape is (1, 577, 768)
        return x


class PatchEmbed(Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = ops.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else ops.Identity()
        self.flatten = ops.Flatten(2)
        self.transpose = ops.Transpose(1, 2)

    def call(self, x):
        """Forward PatchEmbed."""
        x = self.transpose(self.flatten(self.proj(x)))
        x = self.norm(x)
        return x


@ClassFactory.register(ClassType.NETWORK)
class VisionTransformer(Module):
    """Vision Transformer:`An Image is Worth 16x16 Words: Transformers for Image Recognition atScale.

    :param img_size: input image size
    :type img_size: int, tuple
    :param patch_size: patch_size
    :type patch_size: int, tuple
    :param in_chans: number of input channels
    :type in_chans: int
    :param num_classes: number of class for classification head
    :type num_classes: int
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param depth: depth of transformer
    :type depth: int
    :param num_heads: number of attention heads
    :type num_heads: int
    :param mlp_ratio: ration of mlp hidden dim to embedding dim
    :type mlp_ratio: int
    :param qkv_bias: enable biad for qkv if True
    :type qkv_bias: bool
    :param qk_scale: override default qk scale of head_dim ** -0.5 if set
    :type qk_scale: float
    :param representation_size: enable and set representation layer (pre-logits) to this value if set
    :type representation_size:  (Optional[int])
    :param distilled : model includes a distillation token and head as in DeiT models
    :type distilled: bool
    :parm drop_rate: dropout rate
    :type drop_rate: float
    :parm attn_drop_rate : attention dropout rate
    :type attn_drop_rate: float
    :param drop_path_rate: stochastic depth rate
    :type drop_path_rate: float
    :param embed_layer: patch embedding layer
    :type embed_layer: nn.Module
    :parm norm_layer: : normalization layer
    :type norm_layer: nn.Module
    :param weight_init: weight init scheme
    :type weight_init: str
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """Construct the VisionTransformer class."""
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(ops.LayerNorm, eps=1e-6)
        act_layer = act_layer or ops.gelu

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = ops.Parameter(
            ops.Tensor(np.zeros([1, num_patches + self.num_tokens, embed_dim]).astype(np.float32)), name="pos_embed")

        self.cls_token = ops.Parameter(ops.Tensor(np.zeros([1, 1, embed_dim]).astype(np.float32)), name="cls_token")

        self.pos_drop = ops.Dropout(prob=drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = Sequential(OrderedDict([
                ('fc', ops.Linear(embed_dim, representation_size)),
                ('act', ops.Tanh())
            ]))
        else:
            self.pre_logits = ops.Identity()

        # Classifier head(s)
        self.head = ops.Linear(self.num_features, num_classes) if num_classes > 0 else ops.Identity()

    def call(self, x):
        """Forward VisionTransformer."""
        x = self.patch_embed(x)
        cls_token = ops.expand(self.cls_token, (x.shape[0], 1, 1))
        x = ops.concat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0:1, :]
        x = ops.Squeeze(1)(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x
