import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
from aaa import common
from .utils.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding


class PyramidAttention(nn.Module):
    def __init__(self, level=2, res_scale=1, channel=64, reduction=4, ksize=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1 - i / 10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input, small):
        res = input
        # print('small-input',input.shape)
        # theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base, 1, dim=0)
        # patch size for matching
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        # build feature pyramid
        for i in range(len(self.scale)):
            ref = input
            if self.scale[i] != 1:
                ref = small
            # feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            # sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                            strides=[self.stride, self.stride],
                                            rates=[1, 1],
                                            padding='same')  # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            # feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            # sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            # group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
            # normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),self.escape_NaN)
            wi_normed = wi / max_wi
            # matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            # print('small-yi',yi.shape)
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi * self.softmax_scale, dim=1)

            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
            # print('small-pyramid_yishape', yi.shape)
            # print('small-pyramid_shape_raw_wi', raw_wi.shape)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
            y.append(yi)

        y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
        return y

class LargePyramidAttention(nn.Module):
    def __init__(self, level=2, res_scale=1, channel=20, reduction=1, ksize=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(LargePyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1 - i / 10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input, large):
        # print('input-shape',input.shape)
        res = input
        # theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base, 1, dim=0)
        # patch size for matching
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        # build feature pyramid
        for i in range(len(self.scale)):
            ref = large
            if self.scale[i] != 1:
                ref = large
            # feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            # sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                            strides=[self.stride, self.stride],
                                            rates=[1, 1],
                                            padding='same')  # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            # print('raw_w_i',raw_w_i.shape)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
            # print('raw_w_ipppppppp', raw_w_i.shape)
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            # feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            # sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            # group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
            # normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi
            # matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            print('1111111111111111111111111111111111')
            print('xi',xi.shape)
            print('wi_normed',wi_normed.shape)
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi * self.softmax_scale, dim=1)

            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
            print('yi', yi.shape)
            print('raw_wi', raw_wi.shape)
            print('222222222222222222222222222222')
            # raw_wi = raw_wi.permute(1, 0, 2, 3)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
            print('yi-------',yi.shape)
            y.append(yi)

        y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
        return y


class CrossScaleAttention(nn.Module):
    def __init__(self, channel=20, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match_2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        # self.register_buffer('fuse_weight', fuse_weight)

    def forward(self, input, small):
        # get embedding
        match_input = self.conv_match_1(input)
        shape_base = list(match_input.size())  # b*c*h*w
        input_groups = torch.split(match_input, 1, dim=0)

        # b*c*h*w
        embed_w = self.conv_assembly(small)
        shape_input = list(embed_w.size())  # b*c*h*w

        # kernel size on input for matching
        kernel = self.ksize

        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                      strides=[self.stride, self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling X to form Y for cross-scale matching
        ref = small
        ref = self.conv_match_2(ref)
        w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        shape_ref = ref.shape
        # w shape: [N, C, k, k, L]
        w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        y = []
        scale = self.softmax_scale
        # 1*1*k*k
        # fuse_weight = self.fuse_weight

        for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
            # normalize
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi

            # Compute correlation map
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # rescale matching score
            yi = F.softmax(yi * scale, dim=1)
            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for reconsturction
            wi_center = raw_wi[0]
            # print('yi', yi.shape)
            # print('wi_center', wi_center.shape)
            yi = F.conv_transpose2d(yi, wi_center, stride=self.stride * 2, padding=1)

            yi = yi / 6.
            y.append(yi)

        y = torch.cat(y, dim=0)
        return y


# class PyramidAttention(nn.Module):
#     def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
#                  conv=common.default_conv):
#         super(PyramidAttention, self).__init__()
#         self.ksize = ksize
#         self.stride = stride
#         self.res_scale = res_scale
#         self.softmax_scale = softmax_scale
#         self.scale = [1 - i / 10 for i in range(level)]
#         self.average = average
#         escape_NaN = torch.FloatTensor([1e-4])
#         self.register_buffer('escape_NaN', escape_NaN)
#         self.conv_match_L_base = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
#         self.conv_match = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
#         self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
#
#     def forward(self, input):
#         res = input
#         # theta
#         match_base = self.conv_match_L_base(input)
#         shape_base = list(res.size())
#         input_groups = torch.split(match_base, 1, dim=0)
#         # patch size for matching
#         kernel = self.ksize
#         # raw_w is for reconstruction
#         raw_w = []
#         # w is for matching
#         w = []
#         # build feature pyramid
#         for i in range(len(self.scale)):
#             ref = input
#             if self.scale[i] != 1:
#                 ref = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
#             # feature transformation function f
#             base = self.conv_assembly(ref)
#             shape_input = base.shape
#             # sampling
#             raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
#                                             strides=[self.stride, self.stride],
#                                             rates=[1, 1],
#                                             padding='same')  # [N, C*k*k, L]
#             raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
#             raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
#             raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
#             raw_w.append(raw_w_i_groups)
#
#             # feature transformation function g
#             ref_i = self.conv_match(ref)
#             shape_ref = ref_i.shape
#             # sampling
#             w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
#                                         strides=[self.stride, self.stride],
#                                         rates=[1, 1],
#                                         padding='same')
#             w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
#             w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
#             w_i_groups = torch.split(w_i, 1, dim=0)
#             w.append(w_i_groups)
#
#         y = []
#         for idx, xi in enumerate(input_groups):
#             # group in a filter
#             wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
#             # normalize
#             max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
#                                                      axis=[1, 2, 3],
#                                                      keepdim=True)),
#                                self.escape_NaN)
#             wi_normed = wi / max_wi
#             # matching
#             xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
#             yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
#             yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
#             # softmax matching score
#             yi = F.softmax(yi * self.softmax_scale, dim=1)
#
#             if self.average == False:class PyramidAttention(nn.Module):
# #     def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
# #                  conv=common.default_conv):
# #         super(PyramidAttention, self).__init__()
# #         self.ksize = ksize
# #         self.stride = stride
# #         self.res_scale = res_scale
# #         self.softmax_scale = softmax_scale
# #         self.scale = [1 - i / 10 for i in range(level)]
# #         self.average = average
# #         escape_NaN = torch.FloatTensor([1e-4])
# #         self.register_buffer('escape_NaN', escape_NaN)
# #         self.conv_match_L_base = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
# #         self.conv_match = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
# #         self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
# #
# #     def forward(self, input):
# #         res = input
# #         # theta
# #         match_base = self.conv_match_L_base(input)
# #         shape_base = list(res.size())
# #         input_groups = torch.split(match_base, 1, dim=0)
# #         # patch size for matching
# #         kernel = self.ksize
# #         # raw_w is for reconstruction
# #         raw_w = []
# #         # w is for matching
# #         w = []
# #         # build feature pyramid
# #         for i in range(len(self.scale)):
# #             ref = input
# #             if self.scale[i] != 1:
# #                 ref = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
# #             # feature transformation function f
# #             base = self.conv_assembly(ref)
# #             shape_input = base.shape
# #             # sampling
# #             raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
# #                                             strides=[self.stride, self.stride],
# #                                             rates=[1, 1],
# #                                             padding='same')  # [N, C*k*k, L]
# #             raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
# #             raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
# #             raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
# #             raw_w.append(raw_w_i_groups)
# #
# #             # feature transformation function g
# #             ref_i = self.conv_match(ref)
# #             shape_ref = ref_i.shape
# #             # sampling
# #             w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
# #                                         strides=[self.stride, self.stride],
# #                                         rates=[1, 1],
# #                                         padding='same')
# #             w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
# #             w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
# #             w_i_groups = torch.split(w_i, 1, dim=0)
# #             w.append(w_i_groups)
# #
# #         y = []
# #         for idx, xi in enumerate(input_groups):
# #             # group in a filter
# #             wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
# #             # normalize
# #             max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
# #                                                      axis=[1, 2, 3],
# #                                                      keepdim=True)),
# #                                self.escape_NaN)
# #             wi_normed = wi / max_wi
# #             # matching
# #             xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
# #             yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
# #             yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
# #             # softmax matching score
# #             yi = F.softmax(yi * self.softmax_scale, dim=1)
# #
# #             if self.average == False:
# #                 yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()
# #
# #             # deconv for patch pasting
# #             raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
# #             yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
# #             y.append(yi)
# #
# #         y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
# #         return y
#                 yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()
#
#             # deconv for patch pasting
#             raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
#             yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
#             y.append(yi)
#
#         y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
#         return y