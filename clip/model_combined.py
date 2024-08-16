from collections import OrderedDict
from typing import Tuple, Union
import math
from functools import reduce
from operator import mul
import numpy as np
from torch.nn import Conv2d, Dropout
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import cv2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


#!EZCLIP###############################################################################################
class Adapter(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Adapter_new(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.attn = nn.MultiheadAttention(
            D_features,
            16,
        )

    def forward(self, x):
        # x is (BT, HW+1, D)
        x = x + self.attn(x, x, x, need_weights=False, attn_mask=None)[0]
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


def tensor_image_old(attn_weight):
    for i, weight in enumerate(attn_weight):
        heatmap = F.interpolate(
            weight.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print(i)
        cv2.imwrite(
            "/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/test_img/img_%d.png"
            % (i),
            heatmap_colormap,
        )


def tensor_image(attn_weight):
    b, _, _ = attn_weight.shape
    attn_weight = torch.mean(attn_weight, 1)[:, 1:].view(b, 14, 14)
    for i, weight in enumerate(attn_weight[:,]):
        heatmap = F.interpolate(
            weight.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print(i)
        cv2.imwrite(
            "/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/test_img/img_%d.png"
            % (i),
            heatmap_colormap,
        )


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#######################################################################################################
#!ResidualAttentionBlock###############################################################################
class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        dropout=0.0,
        frames=8,
        model_for="image",
    ):
        super().__init__()
        self.config = config
        self.T = frames
        self.model_for = model_for
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        # self.prompt_attn = nn.MultiheadAttention(d_model, n_head,dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        if (
            self.model_for == "image"
            and not self.config.prompt.use
            and self.config.T_Adapter
        ):
            self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.Adapter = Adapter(d_model)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_weight(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

    def forward(
        self,
        x: torch.Tensor,
        T_prompt=None,
        Text_prompt=None,
        layer_num=None,
        return_attention=False,
    ):
        if self.model_for == "image":
            l, bt, d = x.size()
            b = bt // self.T
            if T_prompt is not None:
                b = bt // self.T
                x = x.view(l, b, self.T, d)
                ############### This for weighted mean  #########################
                T_prompt = T_prompt.expand(x.shape[1], -1, -1) + torch.mean(x, 0)
                T_prompt = T_prompt.view(b, self.T, 1, d)
                T_prompt = T_prompt.permute(1, 2, 0, 3).view(self.T, b, d)
                T_prompt = self.drop_path(self.attention(self.ln_1(T_prompt)))
                T_prompt = T_prompt.view(self.T, 1, b, d).permute(1, 2, 0, 3)

                x = torch.cat([x, T_prompt], dim=0)
                x = x.view(l + 1, -1, d)
            #################################################################
            # if self.config.network.Return_Attention and layer_num==11:
            #     return self.attention_weight(self.ln_1(x))
            if return_attention:  # ADDED
                return self.attention_weight(self.ln_1(x))  # ADDED
            ###################################
            if (
                T_prompt is None
                and not self.config.prompt.use
                and self.config.T_Adapter
            ):
                xt = (
                    x.view(l, b, self.T, d)
                    .permute(2, 0, 1, 3)
                    .reshape(self.T, l * b, d)
                )
                xt = self.T_Adapter(self.attention(self.ln_1(xt)))
                xt = (
                    xt.view(self.T, l, b, d)
                    .permute(1, 0, 2, 3)
                    .reshape(l, self.T * b, d)
                )
                x = x + self.drop_path(xt)
            ######################################
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x[:l, :, :]
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x
        if self.model_for == "text":
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x


# TODO: Revised
import torch
from torch import nn
from collections import OrderedDict


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        dropout=0.0,
        frames=8,
        model_for="image",
        add_prompt=False,
        text_layer=False,
        i=0,
        design_details=None,
    ):
        super().__init__()
        self.config = config
        self.T = frames
        self.model_for = model_for
        self.add_prompt = add_prompt
        self.text_layer = text_layer
        self.attn_mask = attn_mask

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

        if model_for == "image" and not config.prompt.use and config.T_Adapter:
            self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.Adapter = Adapter(d_model)

        # VPT shallow prompt configuration
        if i != 0 and self.add_prompt:
            if self.text_layer:
                self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                ctx_vectors = torch.empty(self.n_ctx_text, d_model)
            else:
                self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT_shallow = nn.Parameter(ctx_vectors)

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_weight(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

    def forward(
        self,
        x: torch.Tensor,
        T_prompt=None,
        Text_prompt=None,
        layer_num=None,
        return_attention=False,
    ):
        # Appending learnable tokens if required
        if self.add_prompt:
            if not self.text_layer:
                prefix = x[0 : x.shape[0] - self.n_ctx_visual, :, :]
                visual_context = (
                    self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                )
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text :, :, :]
                textual_context = (
                    self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                )
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        if self.model_for == "image":
            l, bt, d = x.size()
            b = bt // self.T
            if T_prompt is not None:
                x = x.view(l, b, self.T, d)
                T_prompt = T_prompt.expand(x.shape[1], -1, -1) + torch.mean(x, 0)
                T_prompt = T_prompt.view(b, self.T, 1, d)
                T_prompt = T_prompt.permute(1, 2, 0, 3).view(self.T, b, d)
                T_prompt = self.drop_path(self.attention(self.ln_1(T_prompt)))
                T_prompt = T_prompt.view(self.T, 1, b, d).permute(1, 2, 0, 3)

                x = torch.cat([x, T_prompt], dim=0)
                x = x.view(l + 1, -1, d)

            if return_attention:
                return self.attention_weight(self.ln_1(x))

            if (
                T_prompt is None
                and not self.config.prompt.use
                and self.config.T_Adapter
            ):
                xt = (
                    x.view(l, b, self.T, d)
                    .permute(2, 0, 1, 3)
                    .reshape(self.T, l * b, d)
                )
                xt = self.T_Adapter(self.attention(self.ln_1(xt)))
                xt = (
                    xt.view(self.T, l, b, d)
                    .permute(1, 0, 2, 3)
                    .reshape(l, self.T * b, d)
                )
                x = x + self.drop_path(xt)

            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x[:l, :, :]
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x

        if self.model_for == "text":
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        dropout=0.0,
        frames=8,
        model_for="image",
        text_layer=False,
        i=0,
        design_details=None,
    ):
        super().__init__()
        self.config = config
        self.T = frames
        self.model_for = model_for
        self.text_layer = text_layer
        self.attn_mask = attn_mask

        # Initialize attention, layer normalization, MLP, and adapter layers
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        # Adapter layers based on the configuration
        if (
            self.model_for == "image"
            and not self.config.prompt.use
            and self.config.T_Adapter
        ):
            self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.Adapter = Adapter(d_model)

        # MaPLe-specific configurations for compound prompts
        self.compound_prompt_nctx = (
            design_details["maple_length"] if design_details else None
        )
        self.first_layer = i == 0

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(
        self,
        inputs,
        T_prompt=None,
        Text_prompt=None,
        layer_num=None,
        return_attention=False,
    ):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]

        if not self.first_layer and len(compound_prompts_deeper) > 0:
            if not self.text_layer:
                if not (counter > len(compound_prompts_deeper) - 1):
                    prefix = x[0 : x.shape[0] - self.compound_prompt_nctx, :, :]
                    visual_context = compound_prompts_deeper[counter]
                    visual_context = (
                        visual_context.expand(x.shape[1], -1, -1)
                        .permute(1, 0, 2)
                        .half()
                    )
                    x = torch.cat([prefix, visual_context], dim=0)
                    counter += 1
            else:
                if not (counter > len(compound_prompts_deeper) - 1):
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.compound_prompt_nctx :, :, :]
                    textual_context = compound_prompts_deeper[counter]
                    textual_context = (
                        textual_context.expand(x.shape[1], -1, -1)
                        .permute(1, 0, 2)
                        .half()
                    )
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    counter += 1

        if self.model_for == "image":
            l, bt, d = x.size()
            b = bt // self.T

            if T_prompt is not None:
                x = x.view(l, b, self.T, d)
                T_prompt = T_prompt.expand(x.shape[1], -1, -1) + torch.mean(x, 0)
                T_prompt = T_prompt.view(b, self.T, 1, d)
                T_prompt = T_prompt.permute(1, 2, 0, 3).view(self.T, b, d)
                T_prompt = self.drop_path(self.attention(self.ln_1(T_prompt)))
                T_prompt = T_prompt.view(self.T, 1, b, d).permute(1, 2, 0, 3)

                x = torch.cat([x, T_prompt], dim=0)
                x = x.view(l + 1, -1, d)

            if return_attention:
                return self.attention_weight(self.ln_1(x))

            if (
                T_prompt is None
                and not self.config.prompt.use
                and self.config.T_Adapter
            ):
                xt = (
                    x.view(l, b, self.T, d)
                    .permute(2, 0, 1, 3)
                    .reshape(self.T, l * b, d)
                )
                xt = self.T_Adapter(self.attention(self.ln_1(xt)))
                xt = (
                    xt.view(self.T, l, b, d)
                    .permute(1, 0, 2, 3)
                    .reshape(l, self.T * b, d)
                )
                x = x + self.drop_path(xt)

            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x[:l, :, :]
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))

        return [x, compound_prompts_deeper, counter]


#######################################################################################################
#!Transformer##########################################################################################
class Transformer(nn.Module):
    def __init__(
        self,
        config,
        width: int,
        layers: int,
        heads: int,
        no_frame: int = 1,
        patch_size: int = None,
        attn_mask: torch.Tensor = None,
        dropout=None,
        model_for="image",
        prompts_needed=0,
        text_layer=False,
        design_details=None,
    ):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)]
        print("dropout used:{}".format(dropout))

        self.width = width
        self.layers = layers
        self.no_frame = no_frame
        self.model_for = model_for
        self.config = config
        self.design_details = design_details

        current_trainer = design_details["trainer"]

        if model_for == "image":
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, (patch_size, patch_size), 1) + self.width)
            )
            if config.prompt.use:
                num_tokens = config.prompt.num_of_token
                self.prompt_proj = nn.Identity()
                self.prompt_dropout = nn.Dropout(config.prompt.DROPOUT)
                if config.prompt.INITIATION == "random":
                    if config.prompt.DEEP:
                        self.T_prompt_embeddings = nn.Parameter(
                            torch.randn(self.layers, self.no_frame, self.width)
                        )
                    else:
                        self.T_prompt_embeddings = nn.Parameter(
                            torch.randn(1, self.no_frame, self.width)
                        )
                    nn.init.uniform_(self.T_prompt_embeddings.data, -val, val)
                else:
                    raise ValueError("Other initiation scheme is not supported")

        if current_trainer == "IVLP" or current_trainer == "VPT":
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock_IVLP(
                        config,
                        width,
                        heads,
                        attn_mask,
                        dropout=dropout[i],
                        frames=self.no_frame,
                        model_for=self.model_for,
                        add_prompt=True if prompts_needed > i else False,
                        text_layer=text_layer,
                        i=i,
                        design_details=design_details,
                    )
                    for i in range(layers)
                ]
            )
        elif current_trainer == "MaPLe":
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock_MaPLe(
                        config,
                        width,
                        heads,
                        attn_mask,
                        dropout=dropout[i],
                        frames=self.no_frame,
                        model_for=self.model_for,
                        text_layer=text_layer,
                        i=i,
                        design_details=design_details,
                    )
                    for i in range(layers)
                ]
            )
        else:
            assert current_trainer == "CoOp" or current_trainer == "CoCoOp"
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(
                        config,
                        width,
                        heads,
                        attn_mask,
                        dropout=dropout[i],
                        frames=self.no_frame,
                        model_for=self.model_for,
                    )
                    for i in range(layers)
                ]
            )

    def forward(self, x: torch.Tensor):
        trainer = self.design_details["trainer"]
        if trainer == 'MaPLe':
            dtype = x[0].dtype
        else:
            dtype = x.dtype
        if self.model_for == "image":
            if self.config.prompt.use:
                for i, block in enumerate(self.resblocks):
                    if i == 0 or (self.config.prompt.DEEP and i < len(self.resblocks)):
                        x = block(
                            x,
                            T_prompt=self.prompt_dropout(
                                self.prompt_proj(
                                    self.T_prompt_embeddings[i : i + 1, :, :]
                                )
                            ).to(dtype),
                            layer_num=i,
                        )
                    else:
                        x = block(x, T_prompt=None)
                return x
            else:
                return self.resblocks(x)
        if self.model_for == "text":
            return self.resblocks(x)

    def forward_attention(self, x: torch.Tensor):
        trainer = self.design_details["trainer"]
        if trainer == 'MaPLe':
            dtype = x[0].dtype
        else:
            dtype = x.dtype
        if self.model_for == "image":
            if self.config.prompt.use:
                for i, block in enumerate(self.resblocks):
                    if i == 0:
                        x = block(
                            x,
                            T_prompt=self.prompt_dropout(
                                self.prompt_proj(
                                    self.T_prompt_embeddings[i : i + 1, :, :]
                                )
                            ).to(dtype),
                            layer_num=i,
                        )
                    elif self.config.prompt.DEEP:
                        if i < len(self.resblocks) - 1:
                            x = block(
                                x,
                                T_prompt=self.prompt_dropout(
                                    self.prompt_proj(
                                        self.T_prompt_embeddings[i : i + 1, :, :]
                                    )
                                ).to(dtype),
                                layer_num=i,
                            )
                        else:
                            x = block(
                                x,
                                T_prompt=self.prompt_dropout(
                                    self.prompt_proj(
                                        self.T_prompt_embeddings[i : i + 1, :, :]
                                    )
                                ).to(dtype),
                                layer_num=i,
                                return_attention=True,
                            )
                    else:
                        x = block(
                            x,
                            T_prompt=self.prompt_dropout(
                                self.prompt_proj(
                                    self.T_prompt_embeddings[i : i + 1, :, :]
                                )
                            ).to(dtype),
                        )
                return x
            else:
                for i, block in enumerate(self.resblocks):
                    if i < len(self.resblocks) - 1:
                        x = block(x)
                    else:
                        x = block(x, return_attention=True)
                return x
        if self.model_for == "text":
            return self.resblocks(x)


#######################################################################################################
#!VisionTransformer####################################################################################

import torch
import torch.nn as nn
from einops import rearrange

# Assuming LayerNorm and Transformer are defined elsewhere
# from your_module import LayerNorm, Transformer


class VisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        input_resolution: int,
        patch_size: int,
        width: int,
        no_frame: int,
        layers: int,
        heads: int,
        output_dim: int,
        dropout=None,
        joint=False,
        emb_dropout=0.0,
        design_details=None,
    ):
        """
        Initializes the VisionTransformer_Combined model.

        Args:
            config: Configuration object (if any) for the transformer.
            input_resolution (int): Resolution of the input images.
            patch_size (int): Size of each image patch.
            width (int): Width of the transformer.
            layers (int): Number of transformer layers.
            heads (int): Number of attention heads.
            output_dim (int): Dimension of the output embeddings.
            design_details (dict): Details for design configurations, especially for VPT.
            no_frame (int): Number of frames (for temporal data).
            dropout (float): Dropout rate.
            joint (bool): Whether to use joint space-time embeddings.
            emb_dropout (float): Embedding dropout rate.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.num_frame = no_frame
        self.joint = joint
        self.emb_dropout = emb_dropout
        self.config = config

        # Convolutional layer to convert images into patch embeddings
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # Visual Prompt Tuning (VPT) Setup
        if design_details and design_details.get("vision_depth", 0) > 0:
            self.VPT_shallow = True
            n_ctx = design_details["vision_ctx"]  # Number of context tokens
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            self.prompt_till_layer_visual = design_details["vision_depth"]
        else:
            self.VPT_shallow = False
            self.prompt_till_layer_visual = 0

        # Embedding Initializations
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )

        # Temporal Embedding for Video Data
        self.temporal_embedding = nn.Parameter(torch.zeros(1, no_frame, width))

        # Joint Space-Time Embedding
        if joint:
            print("=====using joint space-time====")
            self.time_embedding = nn.Parameter(scale * torch.randn(no_frame, width))

        # Embedding Dropout
        if emb_dropout > 0:
            print(f"Embedding Dropout Rate: {emb_dropout}")
            self.dropout = nn.Dropout(emb_dropout)
        else:
            self.dropout = None

        # Pre-LayerNorm
        self.ln_pre = LayerNorm(width)

        # Transformer tailored for video data
        self.transformer = Transformer(
            config,
            width,
            layers,
            heads,
            no_frame,
            patch_size,
            dropout=dropout,
            model_for="image",
            prompts_needed=self.prompt_till_layer_visual,
            text_layer=False,
            design_details=design_details,
        )

        # Post-LayerNorm and Projection
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the VisionTransformer_Combined model.

        Args:
            x (torch.Tensor): Input tensor representing images or video frames.

        Returns:
            torch.Tensor: Output embeddings after processing.
        """
        # Convert images to patch embeddings
        x = self.conv1(x)  # Shape: [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Shape: [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # Shape: [*, grid ** 2, width]

        # Append class embedding
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # Shape: [*, grid ** 2 + 1, width]

        #####prompt#################################################
        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Incorporate Visual Prompt Tuning (VPT) tokens
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        #####EZCLIP#################################################
        # Temporal Embedding for Video Data
        if self.temporal_embedding is not None:
            n = x.shape[1]
            x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frame)
            x = x + self.temporal_embedding.to(x.dtype)
            x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        # Joint Space-Time Embedding
        if self.joint:
            B = x.shape[0] // self.num_frame
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=self.num_frame)
            x = x + self.time_embedding.to(x.dtype)
            x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=self.num_frame)
            x = torch.cat((cls_tokens, x), dim=1)

        # Apply Embedding Dropout if specified
        if self.emb_dropout > 0 and self.dropout:
            x = self.dropout(x)

        # Pre-LayerNorm
        x = self.ln_pre(x)

        # Transformer Processing
        x = x.permute(1, 0, 2)  # Shape: NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Shape: LND -> NLD

        # Post-LayerNorm and Projection
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward_attention(self, x: torch.Tensor):
        """
        Forward pass focusing on attention mechanisms.

        Args:
            x (torch.Tensor): Input tensor representing images or video frames.

        Returns:
            torch.Tensor: Attention maps or related outputs.
        """
        # Convert images to patch embeddings
        x = self.conv1(x)  # Shape: [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Shape: [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # Shape: [*, grid ** 2, width]

        # Append class embedding
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # Shape: [*, grid ** 2 + 1, width]

        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Temporal Embedding for Video Data
        if self.temporal_embedding is not None:
            n = x.shape[1]
            x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frame)
            x = x + self.temporal_embedding.to(x.dtype)
            x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        # Pre-LayerNorm
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # Shape: NLD -> LND

        # Obtain Attention Maps
        attention_outputs = self.transformer.forward_attention(x)
        return attention_outputs


class VisionTransformer_MaPLe(nn.Module):
    def __init__(
        self,
        config,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        no_frame: int,
        dropout=None,
        joint=False,
        emb_dropout=0.0,
        design_details=None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.VPT_shallow = True
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.temporal_embedding = (
            nn.Parameter(torch.zeros(1, no_frame, width)) if config else None
        )
        self.dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else None
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        self.config = config
        self.num_frame=no_frame

        if joint:
            print("=====using joint space-time====")
            self.time_embedding = nn.Parameter(scale * torch.randn(no_frame, width))

        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(
            config,
            width,
            layers,
            heads,
            no_frame,
            patch_size,
            dropout=dropout,
            model_for="image",
            prompts_needed=self.prompt_till_layer_visual,
            text_layer=False,
            design_details=design_details,
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        if self.temporal_embedding is not None:
            n = x.shape[1]
            x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frame)
            x = x + self.temporal_embedding.to(x.dtype)
            x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        if self.joint:
            B = x.shape[0] // self.num_frame
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=self.num_frame)
            x = x + self.time_embedding.to(x.dtype)
            x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=self.num_frame)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.emb_dropout > 0:
            x = self.dropout(x)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer(
            [x, compound_deeper_prompts, 0]
        )
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward_attention(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.temporal_embedding is not None:
            n = x.shape[1]
            x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frame)
            x = x + self.temporal_embedding.to(x.dtype)
            x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.forward_attention(x)
        return x


#######################################################################################################
#!CLIP#################################################################################################


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        config,
        # Vision parameters
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        no_frame: int,
        # Text parameters
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        joint=False,
        tsm=False,
        T=8,
        dropout=0.0,
        emb_dropout=0.0,
        design_details=None,  # Additional design details for submodules
    ):
        super().__init__()

        self.config = config
        self.context_length = context_length
        self.design_details = design_details

        if dropout > 0.0:
            dpr = [
                x.item() for x in torch.linspace(0, dropout, vision_layers)
            ]  # stochastic depth decay rule
        else:
            dpr = None

        vision_heads = vision_width // 64
        trainer = design_details.get("trainer") if design_details else None

        # Initialize the visual transformer depending on the trainer type
        if trainer == "MaPLe":
            self.visual = VisionTransformer_MaPLe(
                config,
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                no_frame=no_frame,
                joint=joint,
                dropout=dpr,
                emb_dropout=emb_dropout,
                design_details=design_details,
            )
        else:
            self.visual = VisionTransformer(
                config,
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                no_frame=no_frame,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                joint=joint,
                dropout=dpr,
                emb_dropout=emb_dropout,
                design_details=design_details,
            )

        if tsm:
            print("=========using TSM==========")
            from modules.temporal_shift import make_temporal_shift_vit

            make_temporal_shift_vit(self.visual, T)

        # Text transformer
        prompt_till_layer_text = (
            design_details.get("language_depth") if design_details else None
        )
        self.transformer = Transformer(
            config,
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr,
            model_for="text",
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details,
        )

        # Text token embedding and positional embedding
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.init_adapter()

    def init_adapter(self):
        ## Initialize S_Adapter
        for n, m in self.named_modules():
            if "Adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(
    state_dict: dict,
    config,
    design_details=None,
    tsm=False,
    T=8,
    dropout=0.0,
    joint=False,
    emb_dropout=0.0,
    pretrain=True,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )

    model = CLIP(
        embed_dim,
        config,
        # vision
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        T,
        # text
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        joint=joint,
        tsm=tsm,
        T=T,
        dropout=dropout,
        emb_dropout=emb_dropout,
        design_details=design_details,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if tsm:
        for k in list(state_dict.keys()):
            if k.find("conv1") > -1 and k.find("layer") > -1:
                n_k = k.split("conv1.")[0] + "conv1.net." + k.split("conv1.")[1]
                state_dict[n_k] = state_dict.pop(k)
            if k.find("resblocks") > -1 and k.find("visual") > -1:
                tmp = ""
                for i, t_ in enumerate(k.split("resblocks.")[1].split(".")):
                    if i >= 1:
                        tmp += "." + t_

                n_k = (
                    k.split("resblocks.")[0]
                    + "resblocks."
                    + k.split("resblocks.")[1].split(".")[0]
                    + ".net"
                    + tmp
                )
                #                 print(n_k)
                state_dict[n_k] = state_dict.pop(k)

    convert_weights(model)
    if pretrain:
        print("loading clip pretrained model!")
        if joint:  # or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        print("not using full clip pretrained model, only visual!")

        for k in list(state_dict.keys()):
            if not k.find("visual") > -1:
                state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

    return model


#######################################################################################################
