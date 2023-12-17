import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations,product
import math
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
from aaa.attention import PyramidAttention,LargePyramidAttention


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels/2))
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x,y,self.channels*2),device=tensor.device).type(tensor.type())
        emb[:,:,:self.channels] = emb_x
        emb[:,:,self.channels:2*self.channels] = emb_y

        return emb[None,:,:,:orig_ch].repeat(batch_size, 1, 1, 1)

class Gate(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(Gate, self).__init__()
        self.channel = in_channel
        self.sigmoid = nn.Sigmoid()
        self.conv12 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv23 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv34 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x1, x2, x3, x4):
        conv12= self.conv12(x2-x1) + x2

        conv23= self.conv23(x3-conv12) + x3

        conv3 = F.softmax(conv23, dim=1)
        conv34 = self.conv34(torch.mul(conv3, x4)) +x4

        return conv34


class DenseAgrregation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseAgrregation, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        pad1 = int(1 * (3 - 1) / 2)
        pad2 = int(2 * (3 - 1) / 2)
        pad4 = int(4 * (3 - 1) / 2)
        pad8 = int(8 * (3 - 1) / 2)
        if self.in_channel != self.out_channel:
            self.convert1 = nn.Sequential(nn.Conv2d(self.in_channel, self.out_channel, 1, 1),
                                    nn.LeakyReLU(0.2))

        self.conv11 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad1, dilation=1),
                                    nn.LeakyReLU(0.2))
        self.conv12 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad1, dilation=1),
                                    nn.LeakyReLU(0.2))
        self.conv13 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad1, dilation=1),
                                    nn.LeakyReLU(0.2))
        self.gate11 = Gate(self.out_channel, self.out_channel)
        self.gate12 = Gate(self.out_channel, self.out_channel)

        self.conv21 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad2, dilation=2),
                                    nn.LeakyReLU(0.2))
        self.conv22 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad2, dilation=2),
                                    nn.LeakyReLU(0.2))
        self.conv23 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad2, dilation=2),
                                    nn.LeakyReLU(0.2))
        self.gate21 = Gate(self.out_channel, self.out_channel)
        self.gate22 = Gate(self.out_channel, self.out_channel)

        self.conv41 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad4, dilation=4),
                                    nn.LeakyReLU(0.2))
        self.conv42 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad4, dilation=4),
                                    nn.LeakyReLU(0.2))
        self.conv43 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad4, dilation=4),
                                    nn.LeakyReLU(0.2))
        self.gate41 = Gate(self.out_channel, self.out_channel)
        self.gate42 = Gate(self.out_channel, self.out_channel)

        self.conv81 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad8, dilation=8),
                                    nn.LeakyReLU(0.2))
        self.conv82 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad8, dilation=8),
                                    nn.LeakyReLU(0.2))
        self.conv83 = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, 1, padding=pad8, dilation=8),
                                    nn.LeakyReLU(0.2))
        self.gate81 = Gate(self.out_channel, self.out_channel)
        self.gate82 = Gate(self.out_channel, self.out_channel)

        self.fusion = nn.Conv2d(4 * self.out_channel, self.out_channel, 1, 1)


    def forward(self, x):
        ori = x
        if self.in_channel != self.out_channel:
            x = self.convert1(x)
        conv11 = self.conv11(x)
        conv21 = self.conv21(x)
        conv41 = self.conv41(x)
        conv81 = self.conv81(x)
        gate11 = self.gate11(conv11, conv21, conv41, conv81)
        gate21 = self.gate21(conv21, conv11, conv41, conv81)
        gate41 = self.gate41(conv41, conv11, conv21, conv81)
        gate81 = self.gate81(conv81, conv11, conv21, conv41)

        conv12 = self.conv12(gate11)
        conv22 = self.conv22(gate21)
        conv42 = self.conv42(gate41)
        conv82 = self.conv82(gate81)
        gate12 = self.gate12(conv12, conv22, conv42, conv82)
        gate22 = self.gate22(conv22, conv12, conv42, conv82)
        gate42 = self.gate42(conv42, conv12, conv22, conv82)
        gate82 = self.gate82(conv82, conv12, conv22, conv42)

        conv13 = self.conv13(gate12)
        conv23 = self.conv23(gate22)
        conv43 = self.conv43(gate42)
        conv83 = self.conv83(gate82)
        fusion = self.fusion(torch.cat([conv13, conv23, conv43, conv83],dim=1))
        out = x + fusion
        return out

# class Derain_Module(nn.Module):
#     def __init__(self):
#         super(Derain_Module, self).__init__()
#         self.channel = settings.channel
#         self.conv1 = nn.Sequential(DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel))
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Sequential(DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel),
#                                    )
#         self.pool2 = nn.MaxPool2d(2, 2)
#
#         self.conv3 = nn.Sequential(DenseAgrregation(self.channel, self.channel),
#                                    DenseAgrregation(self.channel, self.channel),
#                                    )
#         self.pool3 = nn.MaxPool2d(2, 2)
#
#         self.deconv1 = nn.Sequential(DenseAgrregation(self.channel, self.channel),
#                                      DenseAgrregation(self.channel, self.channel),
#                                      )
#
#         self.deconv2 = nn.Sequential(DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel),
#                                    )
#         self.deconv3 = nn.Sequential(DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel),
#                                    DenseAgrregation(self.channel,self.channel))
#
#         self.similarity = Bottomupupbottom()
#
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         b1, c1, h1, w1 = conv1.size()
#         pool1 = self.pool1(conv1)
#         conv2 = self.conv2(pool1)
#         b2, c2, h2, w2 = conv2.size()
#         pool2 = self.pool2(conv2)
#         conv3 = self.conv3(pool2)
#         b3, c3, h3, w3 = conv3.size()
#         pool3 = self.pool3(conv3)
#
#         similarity = self.similarity(pool3)
#
#         deconv1 = self.deconv1(conv3 + F.upsample(similarity, [h3, w3]))
#         deconv2 = self.deconv2(conv2 + F.upsample(deconv1, [h2, w2]))
#         deconv3 = self.deconv3(conv1 + F.upsample(deconv2, [h1, w1]))
#         return deconv3

class Derain_Module(nn.Module):
    def __init__(self):
        super(Derain_Module, self).__init__()
        self.channel = settings.channel
        self.conv1 = DenseConnectionAdd(12)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DenseConnectionAdd(2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = DenseConnectionAdd(1)

        self.deconv1 = DenseConnectionAdd(1)

        self.deconv2 = DenseConnectionAdd(2)

        self.deconv3 = DenseConnectionAdd(12)

        self.similarity = Bottomupupbottom()

        self.fusion1 = nn.Conv2d(2 * self.channel, self.channel, 1, 1)
        self.fusion2 = nn.Conv2d(2 * self.channel, self.channel, 1, 1)

    def forward(self, x):
        conv1, conv1_feature = self.conv1(x)
        b1, c1, h1, w1 = conv1.size()
        pool1 = self.pool1(conv1)
        conv2, conv2_feature = self.conv2(pool1)
        b2, c2, h2, w2 = conv2.size()
        pool2 = self.pool2(conv2)
        conv3, conv3_feature = self.conv3(pool2)
        # b3, c3, h3, w3 = conv3.size()
        # pool3 = self.pool3(conv3)

        similarity = self.similarity(conv3)

        deconv1, _ = self.deconv1(similarity, conv3_feature)
        deconv2, _ = self.deconv2(self.fusion1(torch.cat([conv2, F.upsample(deconv1, [h2, w2])], dim=1)), conv2_feature)
        deconv3, _ = self.deconv3(self.fusion1(torch.cat([conv1, F.upsample(deconv2, [h1, w1])], dim=1)), conv1_feature)
        return deconv3

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,3,1)
        enc = self.penc(tensor)
        return enc.permute(0,3,1,2)
def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        self.channel_num = settings.channel
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.res = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
        )
    def forward(self, x):
        convert = x
        out = convert + self.res(convert)
        return out

class DenseConnectionAdd(nn.Module):
    def __init__(self, unit_num,channel=settings.channel):
        super(DenseConnectionAdd, self).__init__()
        self.unit_num = unit_num
        self.channel = channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Residual_Block())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x, encoder=None):
        cat = []
        cat.append(x)
        out = x
        feature = []
        for i in range(self.unit_num):
            if encoder is not None:
                tmp = self.units[i](out + encoder[self.unit_num-i-1])
            else:
                tmp = self.units[i](out)
            feature.append(tmp)
            cat.append(tmp)
            out = self.conv1x1[i](torch.cat(cat, dim=1))
        return out, feature
class eca_layer_max(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer_max, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)


class Bottomupupbottom(nn.Module):
    def __init__(self):
        super(Bottomupupbottom, self).__init__()
        self.channel = settings.channel
        # self.bu1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        # self.bu2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        # self.bu3 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        # self.cross = CrossScaleAttention(channel=settings.channel)
        self.l2s1 = PyramidAttention(level=2, res_scale=1, channel=settings.channel)
        self.l2s2 = PyramidAttention(level=2, res_scale=1, channel=settings.channel)
        self.l2s3 = PyramidAttention(level=2, res_scale=1, channel=settings.channel)
        self.s2l1 = PyramidAttention(level=2, res_scale=1, channel=settings.channel)
        self.s2l2 = PyramidAttention(level=2, res_scale=1, channel=settings.channel)
        self.s2l3 = PyramidAttention(level=2, res_scale=1, channel=settings.channel)

        self.conv11 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),
                                    nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p11 = PositionalEncodingPermute2D(settings.channel)
        self.conv12 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p12 = PositionalEncodingPermute2D(settings.channel)
        self.conv13 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p13 = PositionalEncodingPermute2D(settings.channel)
        self.conv14 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p14 = PositionalEncodingPermute2D(settings.channel)

        # self.position = PositionalEncodingPermute2D(settings.channel)
        self.conv21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p21 = PositionalEncodingPermute2D(settings.channel)
        self.conv22 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p22 = PositionalEncodingPermute2D(settings.channel)
        self.conv23 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p23 = PositionalEncodingPermute2D(settings.channel)
        self.conv24 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.Conv2d(self.channel, self.channel , 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_p24 = PositionalEncodingPermute2D(settings.channel)

        self.fusion = nn.Sequential(nn.Conv2d(6 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))

        self.pooling2 = nn.MaxPool2d(2, 2)


    def forward(self, x):
        # posi = self.position(x)
        # print('position',posi.size())
        pool1 = x
        b1, c1, h1, w1 = pool1.size()
        pool2 = self.pooling2(pool1)
        b2, c2, h2, w2 = pool2.size()
        pool3 = self.pooling2(pool2)
        b3, c3, h3, w3 = pool3.size()
        pool4 = self.pooling2(pool3)
        b4, c4, h4, w4 = pool4.size()

        conv11 = self.conv11(torch.cat([pool4, self.conv_p11(pool4)], dim=1))
        bu1 = self.s2l1(pool3,conv11)
        conv12 = self.conv12(torch.cat([bu1, self.conv_p12(bu1)], dim=1))
        # bu2 = self.bu2(torch.cat([F.upsample(conv12, [h2, w2]), pool2], dim=1))
        bu2 = self.s2l1(pool2, conv12)
        conv13 = self.conv13(torch.cat([bu2,self.conv_p13(bu2)], dim=1))
        # bu3 = self.bu3(torch.cat([F.upsample(conv13, [h1, w1]), pool1], dim=1))
        bu3 = self.s2l1(pool1, conv13)
        conv14 = self.conv14(torch.cat([bu3,self.conv_p14(bu3)], dim=1))

        conv21 = self.conv21(torch.cat([conv14, self.conv_p21(conv14)], dim=1))
        ub1 = self.l2s1(conv13, conv21)
        conv22 = self.conv22(torch.cat([ub1, self.conv_p22(ub1)], dim=1))
        # ub2 = self.ub2(torch.cat([conv12, F.upsample(conv22, [h3, w3])], dim=1))
        ub2 = self.l2s1(conv12, conv22)
        conv23 = self.conv23(torch.cat([ub2, self.conv_p23(ub2)], dim=1))
        # ub3 = self.ub3(torch.cat([conv11, F.upsample(conv23, [h4, w4])], dim=1))
        ub3 = self.l2s1(conv12, conv23)
        conv24 = self.conv24(torch.cat([ub3, self.conv_p24(ub3)], dim=1))

        out = self.fusion(torch.cat([pool1, conv14, conv21,
                                     F.upsample(pool4, [h1, w1]),
                                     F.upsample(conv11, [h1, w1]),
                                     F.upsample(conv24, [h1, w1])], dim=1))
        return out
# class Bottomupupbottom(nn.Module):
#     def __init__(self):
#         super(Bottomupupbottom, self).__init__()
#         self.channel = settings.channel
#         self.bu1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#         self.bu2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#         self.bu3 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#         self.cross = CrossScaleAttention(channel=settings.channel)
#         self.large = LargePyramidAttention(level=2, res_scale=1, channel=settings.channel)
#
#         self.conv11 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p11 = nn.Conv2d(2, 2, kernel_size=1)
#         self.conv12 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p12 = nn.Conv2d(2, 2, kernel_size=1)
#         self.conv13 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p13 = nn.Conv2d(2, 2, kernel_size=1)
#         self.conv14 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p14 = nn.Conv2d(2, 2, kernel_size=1)
#
#         self.ub1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#         self.ub2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#         self.ub3 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#         self.pyramid = PyramidAttention(level=2, res_scale=1, channel=settings.channel)
#         self.conv21 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p21 = nn.Conv2d(2, 2, kernel_size=1)
#         self.conv22 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p22 = nn.Conv2d(2, 2, kernel_size=1)
#         self.conv23 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p23 = nn.Conv2d(2, 2, kernel_size=1)
#         self.conv24 = nn.Sequential(nn.Conv2d(self.channel + 2, self.channel , 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv_p24 = nn.Conv2d(2, 2, kernel_size=1)
#
#         self.fusion = nn.Sequential(nn.Conv2d(6 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
#
#         self.pooling2 = nn.MaxPool2d(2, 2)
#
#
#     def forward(self, x):
#         pool1 = x
#         b1, c1, h1, w1 = pool1.size()
#         pool2 = self.pooling2(pool1)
#         b2, c2, h2, w2 = pool2.size()
#         pool3 = self.pooling2(pool2)
#         b3, c3, h3, w3 = pool3.size()
#         pool4 = self.pooling2(pool3)
#         b4, c4, h4, w4 = pool4.size()
#
#         conv11 = self.conv11(torch.cat([pool4, self.conv_p11(position(pool4.shape[2], pool4.shape[3], pool4.is_cuda).repeat(pool4.shape[0], 1, 1, 1))], dim=1))
#         bu1 = self.bu1(torch.cat([F.upsample(conv11, [h3, w3]), pool3], dim=1))
#         conv12 = self.conv12(torch.cat([bu1, self.conv_p12(position(bu1.shape[2], bu1.shape[3], bu1.is_cuda)).repeat(bu1.shape[0], 1, 1, 1)], dim=1))
#         bu2 = self.bu2(torch.cat([F.upsample(conv12, [h2, w2]), pool2], dim=1))
#         conv13 = self.conv13(torch.cat([bu2,self.conv_p13(position(bu2.shape[2], bu2.shape[3], bu2.is_cuda)).repeat(bu2.shape[0], 1, 1, 1)], dim=1))
#         bu3 = self.bu3(torch.cat([F.upsample(conv13, [h1, w1]), pool1], dim=1))
#         conv14 = self.conv14(torch.cat([bu3,self.conv_p14(position(bu3.shape[2], bu3.shape[3], bu3.is_cuda)).repeat(bu3.shape[0], 1, 1, 1)], dim=1))
#
#         conv21 = self.conv21(torch.cat([conv14, self.conv_p21(position(conv14.shape[2], conv14.shape[3], conv14.is_cuda)).repeat(conv14.shape[0], 1, 1, 1)], dim=1))
#         ub1 = self.ub1(torch.cat([conv13, F.upsample(conv21, [h2, w2])], dim=1))
#         # cross = self.cross(conv13, conv21)
#         pyramid = self.pyramid(conv21, conv13)
#         print('111111111111111111111')
#         print('conv13',conv13.size())
#         print('conv21', conv21.size())
#         print('pyramid', pyramid.size())
#         # print('cross',cross.size())
#
#
#         print('bu1', bu1.size())
#         print('conv11',conv11.size())
#         print('pool3', pool3.size())
#         large = self.large(conv11, pool3)
#         print('large', large.size())
#         print('22222222222222222222222')
#         conv22 = self.conv22(torch.cat([ub1, self.conv_p22(position(ub1.shape[2], ub1.shape[3], ub1.is_cuda)).repeat(ub1.shape[0], 1, 1, 1)], dim=1))
#         ub2 = self.ub2(torch.cat([conv12, F.upsample(conv22, [h3, w3])], dim=1))
#         conv23 = self.conv23(torch.cat([ub2, self.conv_p23(position(ub2.shape[2], ub2.shape[3], ub2.is_cuda)).repeat(ub2.shape[0], 1, 1, 1)], dim=1))
#         ub3 = self.ub3(torch.cat([conv11, F.upsample(conv23, [h4, w4])], dim=1))
#         conv24 = self.conv24(torch.cat([ub3, self.conv_p24(position(ub3.shape[2], ub3.shape[3], ub3.is_cuda)).repeat(ub3.shape[0], 1, 1, 1)], dim=1))
#
#         out = self.fusion(torch.cat([pool1, conv14, conv21,
#                                      F.upsample(pool4, [h1, w1]),
#                                      F.upsample(conv11, [h1, w1]),
#                                      F.upsample(conv24, [h1, w1])], dim=1))
#         return out



class MultiViewAggregation(nn.Module):
    def __init__(self):
        super(MultiViewAggregation, self).__init__()
        self.channel = settings.channel
        self.conv3 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 5, 1, 2), nn.LeakyReLU(0.2))

        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool5 = nn.AdaptiveAvgPool2d(1)
        self.conv13 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv15 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.conv351 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv352 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv354 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.conv124 = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.pooling2 = nn.MaxPool2d(2, 2)
        self.pooling4 = nn.MaxPool2d(4, 4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ori = x
        b, c, h, w = x.size()
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv13 = self.conv13(conv3)
        avg_3 = self.sigmoid(self.avg_pool3(conv13))
        conv15 = self.conv15(conv5)
        avg_5 = self.sigmoid(self.avg_pool5(conv15))

        pool31 = conv3
        pool32 = self.pooling2(conv3)
        pool34 = self.pooling4(conv3)

        pool31 = avg_3.expand_as(pool31) * pool31
        pool32 = avg_3.expand_as(pool32) * pool32
        pool34 = avg_3.expand_as(pool34) * pool34

        pool51 = conv5
        pool52 = self.pooling2(conv5)
        pool54 = self.pooling4(conv5)

        pool51 = avg_5.expand_as(pool51) * pool51
        pool52 = avg_5.expand_as(pool52) * pool52
        pool54 = avg_5.expand_as(pool54) * pool54

        conv351 = self.conv351(torch.cat([pool31, pool51], dim=1))
        conv352 = self.conv352(torch.cat([pool32, pool52], dim=1))
        conv354 = self.conv354(torch.cat([pool34, pool54], dim=1))

        conv124 = self.conv124(torch.cat([F.upsample(conv351, [h, w]),
                                          F.upsample(conv352, [h, w]),
                                          F.upsample(conv354, [h, w])], dim=1))

        out = conv124 + ori


        return out
class DenseScaleFusion(nn.Module):
    def __init__(self):
        super(DenseScaleFusion, self).__init__()
        self.channel = settings.channel
        # self.num_scale = settings.num_scale
        self.up_down_fusion1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.up_down_fusion2 = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.up_down_fusion3 = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.down_up_fusion1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.down_up_fusion2 = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.down_up_fusion3 = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.encoder1 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.encoder2 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.encoder3 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.encoder4 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.decoder1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.decoder2 = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.decoder3 = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.decoder4 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.pooling2 = nn.MaxPool2d(2, 2)
        self.pooling4 = nn.MaxPool2d(4, 4)
        self.pooling8 = nn.MaxPool2d(8, 8)

    def forward(self, x):
        ori = x

        encoder1 = self.encoder1(x)
        b1, c1, h1, w1 = encoder1.size()
        pool1 = self.pooling2(encoder1)
        encoder2 = self.encoder2(pool1)
        b2, c2, h2, w2 = encoder2.size()
        pool2 = self.pooling2(encoder2)
        encoder3 = self.encoder3(pool2)
        b3, c3, h3, w3 = encoder3.size()
        pool3 = self.pooling2(encoder3)
        encoder4 = self.encoder4(pool3)
        b4, c4, h4, w4 = encoder4.size()

        down_up_fusion1 = self.down_up_fusion1(torch.cat([F.upsample(encoder4, [h3, w3]), encoder3], dim=1))

        down_up_fusion2 = self.down_up_fusion2(torch.cat([F.upsample(encoder4, [h2, w2]),
                                                          F.upsample(encoder3, [h2, w2]), encoder2], dim=1))
        down_up_fusion3 = self.down_up_fusion3(torch.cat([F.upsample(encoder4, [h1, w1]),
                                                          F.upsample(encoder3, [h1, w1]),
                                                          F.upsample(encoder2, [h1, w1]), encoder1], dim=1))

        up_down_fusion1 = self.up_down_fusion1(torch.cat([F.upsample(encoder1, [h2, w2]), encoder2], dim=1))
        up_down_fusion2 = self.up_down_fusion2(torch.cat([F.upsample(encoder2, [h3, w3]),
                                                          F.upsample(encoder3, [h3, w3]), encoder3], dim=1))
        up_down_fusion3 = self.up_down_fusion3(torch.cat([F.upsample(encoder2, [h4, w4]),
                                                          F.upsample(encoder3, [h4, w4]),
                                                          F.upsample(encoder4, [h4, w4]), encoder4], dim=1))

        decoder1 = self.decoder1(torch.cat([up_down_fusion3, encoder4], dim=1))
        decoder2 = self.decoder2(torch.cat([F.upsample(decoder1, [h3, w3]), up_down_fusion2, down_up_fusion1], dim=1))
        decoder3 = self.decoder3(torch.cat([F.upsample(decoder2, [h2, w2]), up_down_fusion1, down_up_fusion2], dim=1))
        decoder4 = self.decoder4(torch.cat([F.upsample(decoder3, [h1, w1]), down_up_fusion3], dim=1))

        out = decoder4 + ori
        return out


class DenseConnection(nn.Module):
    def __init__(self, unit_num):
        super(DenseConnection, self).__init__()
        self.unit_num = unit_num
        self.channel = settings.channel
        self.units1 = nn.ModuleList()
        self.conv1x11 = nn.ModuleList()
        self.units2 = nn.ModuleList()
        self.conv1x12 = nn.ModuleList()
        self.buub = Bottomupupbottom()
        for i in range(self.unit_num):
            self.units1.append(Residual_Block())
            self.conv1x11.append(nn.Sequential(nn.Conv2d((i+2)*self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))
        for i in range(self.unit_num):
            self.units2.append(Residual_Block())
            self.conv1x12.append(nn.Sequential(nn.Conv2d((i+2)*self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))
    
    def forward(self, x):
        cat = []
        cat.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units1[i](out)
            cat.append(tmp)
            out = self.conv1x11[i](torch.cat(cat, dim=1))

        buub =self.buub(out)

        cat2 = []
        cat2.append(buub)
        out2 = buub
        for i in range(self.unit_num):
            tmp = self.units2[i](out2)
            cat2.append(tmp)
            out = self.conv1x12[i](torch.cat(cat2, dim=1))

        return out+x


# class DenseConnectionAdd(nn.Module):
#     def __init__(self, unit_num):
#         super(DenseConnectionAdd, self).__init__()
#         self.unit_num = unit_num
#         self.channel = settings.channel
#         self.units1 = nn.ModuleList()
#         self.units2 = nn.ModuleList()
#         self.conv1x12 = nn.ModuleList()
#         self.buub = Bottomupupbottom()
#         for i in range(self.unit_num):
#             self.units1.append(Residual_Block())
#             self.conv1x11.append(
#                 nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))
#         for i in range(self.unit_num):
#             self.units2.append(Residual_Block())
#             self.conv1x12.append(
#                 nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))
#
#     def forward(self, x):
#         cat = []
#         cat.append(x)
#         out = x
#         for i in range(self.unit_num):
#             tmp = self.units1[i](out)
#             cat.append(tmp)
#             out = self.conv1x11[i](torch.cat(cat, dim=1))
#
#         buub = self.buub(out)
#
#         cat2 = []
#         cat2.append(buub)
#         out2 = buub
#         for i in range(self.unit_num):
#             tmp = self.units2[i](out2)
#             cat2.append(tmp)
#             out = self.conv1x12[i](torch.cat(cat2, dim=1))
#
#         return out + x

class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        self.channel = settings.channel
        self.unit_num = settings.unit
        self.enterBlock1 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.enterBlock2 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.enterBlock4 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.enterBlock2_2 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.multienter2 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.multienter4 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.multiexit2 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))
        self.multiexit4 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))

        self.net11 = Derain_Module()
        self.net12 = Derain_Module()
        self.net13 = Derain_Module()

        # self.net11 = Residual_Block()
        # self.net12 = Residual_Block()
        # self.net13 = Residual_Block()

        self.fusion11 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.fusion12 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.exitBlock1 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))

        self.net21 = Derain_Module()
        self.net22 = Derain_Module()
        # self.net21 = Residual_Block()
        # self.net22 = Residual_Block()
        self.up2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.exitBlock2 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))

        self.net31 = Derain_Module()
        # self.net31 = Residual_Block()
        self.up3 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.exitBlock3 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))

    def forward(self, x, x2, x4):
        enterBlock1 = self.enterBlock1(x)

        enterBlock2 = self.enterBlock2(x)
        enterBlock3 = self.enterBlock4(x)

        net11 = self.net11(enterBlock1)

        net21 = self.net21(self.up2(torch.cat([net11, enterBlock2], dim=1)))

        net31 = self.net31(self.up3(torch.cat([net21, enterBlock3], dim=1)))

        net12 = self.net12(self.fusion11(torch.cat([net21, net11], dim=1)))

        net22 = self.net22(self.fusion21(torch.cat([net31, net21], dim=1)))

        net13 = self.net31(self.fusion12(torch.cat([net22, net12], dim=1)))

        out1 = x - self.exitBlock1(net13)
        out2 = x - self.exitBlock2(net22)
        out3 = x - self.exitBlock3(net31)

        multienter2 = self.multienter2(x2)
        net11_2 = self.net11(multienter2)
        enterBlock2_2 = self.enterBlock2_2(x2)
        net21_2 = self.net21(self.up2(torch.cat([net11_2, enterBlock2_2], dim=1)))

        multiexit2 = x2 - self.multiexit2(self.net12(net21_2))
        multiexit4 = x4 - self.multiexit4(self.net11(self.multienter4(x4)))

        return out1, out2, out3, multiexit2, multiexit4
class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(1, 3, 5, 9, 13), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features