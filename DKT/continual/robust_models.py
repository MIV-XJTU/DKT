import copy

import torch
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import _cfg
from einops import rearrange

from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        if in_features == 768:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        else:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
            self.bn3 = nn.BatchNorm2d(out_features)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.in_features == 768:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            B, N, C = x.shape
            x = x.reshape(B, int(N ** 0.5), int(N ** 0.5), C).permute(0, 3, 1, 2)
            x = self.bn1(self.fc1(x))
            x = self.act(x)
            x = self.drop(x)
            x = self.act(self.bn2(self.dwconv(x)))
            x = self.bn3(self.fc2(x))
            x = self.drop(x)
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, :64]).reshape(B, 64, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 64, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_CA=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_CA:
            self.attnCA = SKAB(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                use_mask=False)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.use_CA = use_CA
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, tokens):
        if self.use_CA:
            x_cls = x[:, :1]
            x_cls = x_cls + self.drop_path(self.attnCA(self.norm1(x)))
            x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
            return x_cls
        if tokens is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:  # use the GKAB
            x = x + self.drop_path(self.attn(self.norm1(torch.cat((x, tokens), dim=1))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SKAB(nn.Module):

    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias)
        self.k = fc(dim, dim, bias=qkv_bias)
        self.v = fc(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mode=False, attn_mask=False, nb_task_tokens=1, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:, :nb_task_tokens]).reshape(B, nb_task_tokens, self.num_heads, C // self.num_heads).permute(0, 2,
                                                                                                                  1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, nb_task_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, use_mask=False, masked_block=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_CA=False
            )
            for i in range(depth)])
        self.blocks.append(Block(
            dim=embed_dim,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_prob[0],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_CA=True
        ))

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.permute(0,2,3,1).reshape(B, H * W, C)
        for i in range(self.depth):
            x = self.blocks[i](x)
        x = self.blocks[-1](x)
        # x = x.reshape(B, H, W, C).permute(0,3,1,2)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):
        x = self.conv(x)

        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super(conv_embedding, self).__init__()

        self.out_channels = out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(4, 4), stride=(4, 4))

        )
        self.proj1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.proj2 = nn.BatchNorm2d(32)
        self.proj3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.proj4 = nn.Conv2d(32, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.proj3(x)
        x = self.proj4(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, use_mask=False, masked_block=None):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size / stride))

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.base_dims[0] * heads[0]))
        trunc_normal_(self.cls_token, std=.02)

        self.patch_size = patch_size
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            if stage == 0:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob, use_mask=use_mask, masked_block=masked_block)
                )
            else:
                self.transformers.append(
                    Transformer(base_dims[stage], depth[stage], heads[stage],
                                mlp_ratio,
                                drop_rate, attn_drop_rate, drop_path_prob)
                )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        cls_features = self.norm(self.gap(x).squeeze())

        return cls_features

    def forward(self, x):
        cls_features = self.forward_features(x)
        output = self.head(cls_features)
        return output


@register_model
def rvt_tiny(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
            torch.load('rvt_ti.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model


@register_model
def rvt_tiny_plus(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=10,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
            torch.load('rvt_ti*.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model


@register_model
def rvt_small(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[6],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
            torch.load('rvt_small.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model


@register_model
def rvt_small_plus(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[6],
        mlp_ratio=4,
        use_mask=True,
        masked_block=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
            torch.load('rvt_small*.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model


@register_model
def rvt_base(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[12],
        mlp_ratio=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
            torch.load('rvt_base.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model


@register_model
def rvt_base_plus(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[64],
        depth=[12],
        heads=[12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=5,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = \
            torch.load('rvt_base*.pth', map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model
