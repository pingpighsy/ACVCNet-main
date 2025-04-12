# coding=utf-8
from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from senet import senet154
from typing import Optional, Tuple
import math


class DilateAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, dilation: int):
        super(DilateAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.dilation = dilation

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        # Linear projections
        queries = self.query_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                                           2)  # [batch, heads, seq, head_dim]
        keys = self.key_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用膨胀
        dilated_keys = keys[:, :, ::self.dilation, :]
        dilated_values = values[:, :, ::self.dilation, :]

        # 注意力分数
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, dilated_keys) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        # 注意力输出
        context = torch.einsum("bhqk,bhvd->bhqd", attention, dilated_values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        return self.out_proj(context)


class ViTBlock(nn.Module):
    """
        参考了CVNet建立vit模型,使用了空间和通道注意力机制
        Args:
            embed_dim: 嵌入维度，输入张量的通道数
            ffn_latent_dim: ffn的输出维度
            dropout: 整个模型的dropout比例
            ffn_dropout: 前馈神经网络中的dropout比例
            norm_layer: 正则化层的类型，layer_norm_2d
            num_scales: 金字塔层数
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            num_heads: int = 8,
            num_scales: int = 4,
            dropout: Optional[float] = 0.1,
            ffn_dropout: Optional[float] = 0.0,
            norm_layer: Optional[str] = "layer_norm_2d",
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.norm_name = norm_layer
        self.num_scales = num_scales
        self.img_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # 创建一个可训练的位置嵌入
        def get_sinusoidal_positional_embedding(num_positions, embed_dim):
            position = torch.arange(num_positions, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
            pos_embed = torch.zeros(num_positions, embed_dim)
            pos_embed[:, 0::2] = torch.sin(position * div_term)
            pos_embed[:, 1::2] = torch.cos(position * div_term)
            return pos_embed

        self.pos_embed = nn.Parameter(get_sinusoidal_positional_embedding(self.num_patches + 1, embed_dim),
                                      requires_grad=False)
        # 创建线性投影层
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = nn.Linear(self.patch_size * self.patch_size * self.in_channels, embed_dim)

        # Dilate Attention
        self.self_attn = DilateAttention(num_heads, embed_dim, dilation=2)

        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_latent_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_latent_dim, embed_dim),
            nn.Dropout(ffn_dropout)
        )

        # 正则化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 多尺度注意力模块
        self.scale_attns = nn.ModuleList([
            DilateAttention(num_heads, embed_dim, dilation=2),
            DilateAttention(num_heads, embed_dim, dilation=3)
        ])

    def forward(
            self,
            x: Tensor,
            x_prev: Optional[Tensor] = None,
            *args,
            **kwargs
    ) -> Tensor:

        # 将输入图像转换为 token
        b, c, h, w = x.shape
        batch_size = x.size(0)
        # 将图像分割为patch
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.in_channels, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
                                                                self.patch_size * self.patch_size * self.in_channels)

        # 展平
        patches = patches.flatten(start_dim=2)

        # 线性投影
        tokens = self.patch_embed(patches)

        # 添加分类token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # 添加位置嵌入
        tokens = tokens + self.pos_embed
        tokens_fv = tokens
        # 使用膨胀注意力机制
        tokens = tokens.permute(1, 0, 2)  # [num_patches + 1, batch_size, embed_dim]
        tokens = self.self_attn(tokens)  # 这里替换成膨胀注意力
        tokens = tokens.permute(1, 0, 2)  # [batch_size, num_patches + 1, embed_dim]
        tokens = self.norm1(tokens)

        # dropout
        tokens = self.dropout(tokens)

        # 前馈神经网络
        tokens = tokens + self.ffn(tokens)
        tokens = self.norm2(tokens)

        # 提取分类token
        cls_token = tokens[:, 0]

        # 多尺度特征提取
        multi_scale_features = []
        multi_scale_features.append(tokens)
        for scale_attn in self.scale_attns:
            scaled_tokens = scale_attn(tokens_fv)
            multi_scale_features.append(scaled_tokens)

        # 双向特征融合
        fused_features = []
        for i in range(len(multi_scale_features)):
            if i == 0:
                fused_features.append(multi_scale_features[i])
            else:
                prev_feature = F.interpolate(fused_features[-1], size=multi_scale_features[i].shape[2:], mode='linear',
                                             align_corners=True)
                fused_features.append(prev_feature + multi_scale_features[i])

        # 选择最终融合结果
        tokens = fused_features[0]

        # 将 ViT 输出转换为特征图
        tokens = tokens[:, 1:]  # 移除分类 token
        tokens = tokens.view(batch_size, self.num_patches, self.embed_dim)

        # 重塑为特征图
        # (batch_size, height, width, channels)
        feature_map = tokens.permute(0, 2, 1).reshape(batch_size, h // self.patch_size, w // self.patch_size,
                                                      self.embed_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)
        return feature_map


class TopDownFusion(nn.Module):
    def __init__(self, out_channels):
        super(TopDownFusion, self).__init__()

    def forward(self, residual_out, dense_out):
        top_down = F.interpolate(residual_out, scale_factor=2, mode='bilinear', align_corners=False) + dense_out
        return top_down


class BottomUpFusion(nn.Module):
    def __init__(self, out_channels):
        super(BottomUpFusion, self).__init__()

    def forward(self, dense_out, residual_out):
        bottom_up = F.max_pool2d(dense_out, kernel_size=2, stride=2) + residual_out
        return bottom_up


class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.weights = nn.Parameter(torch.ones(2))

    def forward(self, top_down, bottom_up):
        weights = torch.softmax(self.weights, dim=0)
        fused_features = weights[0] * top_down + weights[1] * bottom_up
        return fused_features


class MultiScaleFusion(nn.Module):
    def __init__(self, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.top_down_fusion = TopDownFusion(out_channels)
        self.bottom_up_fusion = BottomUpFusion(out_channels)
        self.weighted_sum = WeightedSum()

    def forward(self, dense_out, residual_out):
        top_down = self.top_down_fusion(residual_out, dense_out)
        bottom_up = self.bottom_up_fusion(dense_out, residual_out)
        fused_features = self.weighted_sum(top_down, bottom_up)
        return fused_features


class DRNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2):
        super(DRNBlock, self).__init__()
        # 瓶颈层
        bottleneck_channels = out_channels // 4
        self.bottleneck_dense = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bottleneck_residual = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)

        # 使用膨胀卷积
        self.depthwise_dense = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=bottleneck_channels),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        )
        self.depthwise_residual = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=bottleneck_channels),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        )

        # 批量归一化层
        self.bn_dense = nn.BatchNorm2d(out_channels)
        self.bn_residual = nn.BatchNorm2d(out_channels)

        # Dropout层
        self.dropout = nn.Dropout2d(0.2)

        # 激活函数
        self.activation = nn.ReLU(inplace=True)

        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFusion(out_channels)

        # 如果输入和输出通道数不匹配，使用1x1卷积进行通道数转换
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 瓶颈层
        x_dense = self.activation(self.bottleneck_dense(x))
        dense_out = self.activation(self.bn_dense(self.depthwise_dense(x_dense)))

        x_residual = self.activation(self.bottleneck_residual(x))
        residual_out = self.activation(self.bn_residual(self.depthwise_residual(x_residual)))

        # Dropout
        dense_out = self.dropout(dense_out)
        residual_out = self.dropout(residual_out)

        # 特征融合
        residual_out = self.multi_scale_fusion(dense_out, residual_out)
        fused_features = self.bi_directional_fusion(dense_out, residual_out)

        # 残差连接
        shortcut = self.shortcut(x)
        output = fused_features + shortcut

        return output


class ConvBlock(nn.Module):
    """Basic convolutional block with DPN.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)
        self.drn_block = DRNBlock(in_c, out_c, k, s, p)

    def forward(self, x):
        return F.relu(self.drn_block(self.bn(self.conv)))



class CrossAttn(nn.Module):
    def __init__(self, embed_dim, out_channels, in_channels_cnn, in_channels_vit, nhead= 32):
        super(CrossAttn, self).__init__()
        self.c = out_channels
        # CNN的输出映射到Query
        self.nhead = nhead
        self.query_conv = nn.Linear(in_channels_cnn, out_channels)
        # ViT的输出映射到Key和Value
        self.key_conv = nn.Linear(in_channels_vit, out_channels)
        self.value_conv = nn.Linear(in_channels_vit, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        # 动态可学习的权重
        self.weights = nn.Parameter(torch.tensor(0.5))
        # 全连接层用于恢复原始形状
        self.fc = nn.Linear(out_channels , out_channels) #你之前这的fc层都没用
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x_cnn, x_vit):
        # CNN->ViT
        # 映射到Query, Key, Value
        b, c, h, w = x_cnn.size() #[b, h, w, c] 你这里的x_cnn与x_vit的大小一样
        x_cnn = x_cnn.view(b, c, -1).permute(0, 2, 1)#(b, hw, c)
        x_vit = x_vit.view(b, c, -1).permute(0 ,2, 1)
        Q1 = self.query_conv(x_cnn) #(b, hw, c)
        K1 = self.key_conv(x_vit)  #b(b, hw, c)
        V1 = self.value_conv(x_vit) #(b, hw, c)

        # ViT->CNN
        Q2 = self.query_conv(x_vit) #(b, hw. c)
        K2 = self.key_conv(x_cnn) #(b, hw, c)
        V2 = self.value_conv(x_cnn) #(b, hw, c)
        Q1 = Q1.view(b, h*w, self.nhead, self.c // self.nhead).permute(0, 2, 1, 3) #[b, nhead, hw, c//nhead]
        Q2 = Q2.view(b, h*w, self.nhead, self.c // self.nhead).permute(0, 2, 1, 3)
        K1 = K1.view(b, h*w, self.nhead, self.c // self.nhead).permute(0, 2, 1, 3)
        V1 = V1.view(b, h*w, self.nhead, self.c // self.nhead).permute(0, 2, 1, 3)
        K2 = K2.view(b, h*w, self.nhead, self.c // self.nhead).permute(0, 2, 1, 3)
        V2 = V2.view(b, h*w, self.nhead, self.c // self.nhead).permute(0, 2, 1, 3)
        # 单模态分数计算
        
        attn_scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) / (self.embed_dim ** 0.5) #(b, nhead, hw, hw)
        attn_scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / (self.embed_dim ** 0.5) #(b, nhead, hw, hw)

        # 将动态权重和交叉注意力机制得分取平均再送入softmax保证权重和为1
        attn_weights1 = self.softmax(attn_scores1) #(b,nhead,  hw, hw)
        attn_weights2 = self.softmax(attn_scores2) #(b, nhead,hw, hw)

        # 公式
        attended_vit_output = torch.matmul(attn_weights1, V1) #[b, hw, hw]  与 [b, hw, c]矩阵相乘 输出 [b, hw, c]
        attended_cnn_output = torch.matmul(attn_weights2, V2)
        attended_vit_output = attended_vit_output.permute(0, 2, 1, 3).contiguous().view(b, h*w, self.c)
        attended_cnn_output = attended_cnn_output.permute(0, 2, 1, 3).contiguous().view(b, h*w, self.c)

        restored_output = attended_cnn_output * self.weights + attended_vit_output * (1 - self.weights)
        restored_output = self.norm(restored_output)
        restored_output = self.fc(restored_output)
        # print(restored_output.size())
        restored_output = restored_output.permute(0, 2, 1).view(b ,self.c , h, w)

        
        return restored_output


class SENet_BASE(nn.Module):

    def __init__(
            self,
            num_classes,
            senet154_weight
    ):
        super(SENet_BASE, self).__init__()
        self.senet154_weight = senet154_weight
        self.num_classes = num_classes

        # construct SEnet154
        senet154_ = senet154(num_classes=1000, pretrained=None)
        senet154_.load_state_dict(torch.load(self.senet154_weight))
        self.main_network = nn.Sequential(senet154_.layer0, senet154_.layer1,
                                          senet154_.layer2, senet154_.layer3,
                                          senet154_.layer4,
                                          nn.AdaptiveAvgPool2d((1, 1)))

        self.global_out = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.main_network(x)
        x = x.view(x.size(0), -1)
        return self.global_out(x)


class MODEL(nn.Module):
    def __init__(self,
                 num_classes,
                 senet154_weight,
                 nchannels=[256, 512, 1024, 2048],
                 embed_dim=1024,
                 ffn_latent_dim=3072,
                 multi_scale=False):
        super(MODEL, self).__init__()
        # self.senet154_weight = senet154_weight
        self.multi_scale = multi_scale
        self.num_classes = num_classes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nchannels[3], num_classes)

        # 构建 SEnet154
        senet154_ = senet154(num_classes=1000, pretrained=None)
        # senet154_.load_state_dict(torch.load(self.senet154_weight))
        self.global_layers = nn.Sequential(
            senet154_.layer0,
            senet154_.layer1,
            senet154_.layer2,
            senet154_.layer3,
            senet154_.layer4,
        )
        self.ha_layers = nn.Sequential(
            CrossAttn(embed_dim, nchannels[3], nchannels[2], nchannels[2]),
            CrossAttn(embed_dim, nchannels[2], self.num_classes, self.num_classes),
            CrossAttn(embed_dim, nchannels[3], self.num_classes, self.num_classes),
        )
        self.vit_block = ViTBlock(embed_dim, ffn_latent_dim)
        self.norm = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.2)  # 添加 self.dropout
        self.global_out = nn.Sequential(
            self.dropout,  # 使用 self.dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        结合 ViT 和 CNN 的前向传播
        """

        def acvc_block(inp, main_conv, harm_attn, vit_output):
            inp = main_conv(inp)

            """# 对cnn输出和vit输出各自做全局平均池
            inp = self.avg_pool(inp)
            vit_output = self.avg_pool(vit_output)

            # 展平
            inp = inp.view(inp.size(0), -1)
            vit_output = vit_output.view(vit_output.size(0), -1)
            cnn = inp
            vit = vit_output

            # 输入至全连接层
            inp = self.fc(inp)
            vit_output = self.fc(vit_output)"""
            
           
            # inp = inp.permute(0, 2, 3, 1)
            
            
            # vit_output = vit_output.permute(0, 2, 3, 1)
            
            # 使用 vit_output 和 inp 作为输入在通道维度进行交叉注意力
            attn_output = harm_attn(inp, vit_output)
            # attn_output = attn_output.permute(0, 3, 1, 2)
          

            attn_output = self.avg_pool(attn_output)
            attn_output = attn_output.view(attn_output.size(0), -1)
            attn_output = self.dropout(attn_output)
            attn_output = self.fc(attn_output)

            # 归一化
            attn_output = self.norm(attn_output)
           
            return attn_output

        # ViT 输入
        # x.size():torch.Size([16, 3, 224, 224])
        vit_input = x
        vit_output = self.vit_block(vit_input)
    

        # CNN 输入
        x = self.global_layers[0](x)
        x1 = self.global_layers[1](x)
        x2 = self.global_layers[2](x1)

        # 512*28*28
        x4 = acvc_block(x2, self.global_layers[3], self.ha_layers[0], vit_output)

        # 全局 pooling
        # global_out = self.global_out(x4)
        global_out = x4
        return global_out