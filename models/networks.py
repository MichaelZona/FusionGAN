import torch
import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np

import complex_net.cmplx_blocks as unet_cmplx
from configs import config
from complex_net.cmplx_blocks import batch_norm
import scipy.io as sio


###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1 and classname.find('RadialBatchNorm') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'uresnet':
        netG = UResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'self_uresnet':
        netG = SelfUResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'stack_self_uresnet':
        netG = StackSelfUResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=6,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'k_stack_self_uresnet':
        netG = KStackSelfUResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=9,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'attention_k_stack_self_uresnet':
        netG = AKStackSelfUResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=9,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 's_k_stack_self_uresnet':
        netG = SAKStackSelfUResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=6,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 's_k_stack_self_uresnet_no_stack':
        netG = SAKStackSelfUResnetGeneratorWithoutStack(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=6,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 's_k_stack_self_uresnet_no_k':
        netG = SAKStackSelfUResnetGeneratorWithoutK(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=6,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 's_k_stack_self_uresnet_no_s':
        netG = SAKStackSelfUResnetGeneratorWithoutS(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=6,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'knet':
        netG = KNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=6,
                                    gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class KNet(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(KNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)


    def forward(self, input,pre,post):
        k_x = torch.fft.fft2(input).cuda()
        real = torch.unsqueeze(torch.real(k_x), -1)
        imag = torch.unsqueeze(torch.imag(k_x), -1)
        k_x = torch.cat((real, imag), -1)

        x0 = k_x

        x1 = self.inc(k_x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        k_x = self.up2(x4, x3)
        k_x = self.up3(k_x, x2)
        k_x = self.up4(k_x, x1)
        k_x = k_x + x0 if config.unet_global_residual_conn else k_x
        k_x = self.ouc(k_x)

        k_real, k_imag = torch.unbind(k_x, dim= -1)
        k_x = torch.complex(k_real, k_imag)

        content = torch.real(torch.fft.ifft2(k_x))
        return content

# class KNet(nn.Module):
#     def __init__(
#             self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
#             n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
#         assert (n_blocks >= 0)
#         super(KNet, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         self.gpu_ids = gpu_ids
#         self.use_parallel = use_parallel
#         self.learn_residual = learn_residual
#
#         mask = sio.loadmat('datasets/1Duniform2.98_ac29.mat')
#         mask = mask['mask']
#         mask = np.fft.ifftshift(mask)
#         mask = torch.from_numpy(mask)
#         mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask, 0), 0), -1)
#         self.mask = torch.cat((mask, mask), -1).cuda()
#
#
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         self.conv1 = unet_cmplx.complexMRIConv(input_nc, 32, True)
#         self.conv2 = unet_cmplx.complexMRIConv(32, 32, True)
#         self.conv3 = unet_cmplx.complexMRIConv(32, 32, True)
#         self.conv4 = unet_cmplx.complexMRIConv(32, 32, True)
#         self.conv5 = unet_cmplx.complexMRIConv(32, output_nc, False)
#
#
#     def forward(self, input,pre,post):
#         k_x = torch.fft.fft2(input).cuda()
#         real = torch.unsqueeze(torch.real(k_x), -1)
#         imag = torch.unsqueeze(torch.imag(k_x), -1)
#         k_x = torch.cat((real, imag), -1)
#
#         x_k = k_x
#         temp = k_x
#
#         # radial_normalizer = batch_norm(
#         #     in_channels=config.in_channels,
#         # ).cuda()
#         # k_x = radial_normalizer(k_x)
#
#         for i in range(10):
#             conv1 = self.conv1(temp)
#             conv2 = self.conv2(conv1)
#             conv3 = self.conv3(conv2)
#             conv4 = self.conv4(conv3)
#             conv5 = self.conv5(conv4)
#
#             # conv_real, conv_imag = torch.unbind(conv5, dim=-1)
#             # conv5 = torch.complex(conv_real, conv_imag)
#             # conv5 = torch.real(torch.fft.ifft2(conv5))
#             #
#             # temp_real, temp_imag = torch.unbind(temp, dim=-1)
#             # temp = torch.complex(temp_real, temp_imag)
#             # s_temp = torch.real(torch.fft.ifft2(temp))
#             #
#             # block = conv5 + s_temp
#             #
#             # k_block = torch.fft.fft2(block)
#             # real = torch.unsqueeze(torch.real(k_block), -1)
#             # imag = torch.unsqueeze(torch.imag(k_block), -1)
#             # k_block = torch.cat((real, imag), -1)
#             #
#             # temp = k_block
#
#             temp = conv5 + temp
#
#         k_real, k_imag = torch.unbind(temp, dim= -1)
#         k_x = torch.complex(k_real, k_imag)
#
#         content = torch.real(torch.fft.ifft2(k_x))
#         return content
class SAKStackSelfUResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SAKStackSelfUResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32+64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_0 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            CBAM(256)
        )


        model = []
        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            CBAM(64)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        # complex conv
        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)

        #fusion
        self.final_conv = nn.Conv2d(2, 1, 1)

    def forward(self, input,pre,post, getKPredict = False, getSPredict = False):
        k_x = torch.fft.fftshift(torch.fft.fft2(input))
        real = torch.unsqueeze(k_x.real, -1)
        imag = torch.unsqueeze(k_x.imag, -1)
        k_x = torch.cat((real, imag), -1)

        x0 = k_x
        x1 = self.inc(k_x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        k_x = self.up2(x4, x3)
        k_x = self.up3(k_x, x2)
        k_x = self.up4(k_x, x1)
        k_x = self.ouc(k_x)
        k_x = x0 + k_x

        k_real, k_imag = torch.unbind(k_x, dim=-1)
        k_x = torch.complex(k_real, k_imag)
        s_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(k_x)))
        if getKPredict:
            return {
                's': s_x,
                'k': k_x
            }

        s_x0 = s_x
        s_x = self.conv1(s_x)
        s_x = self.down_conv1_0(s_x)
        s_x1 = s_x
        s_x = self.down_conv2(s_x)
        s_x2 = s_x
        s_x = self.res_net(s_x)
        s_x = torch.cat([s_x, s_x2], 1)
        s_x = self.up_conv1(s_x)
        s_x = torch.cat([s_x, s_x1], 1)
        s_x = self.up_conv2(s_x)
        s_x = self.conv2(s_x)
        s_x = s_x0 + s_x
        k_x = torch.fft.fftshift(torch.fft.fft2(s_x))
        if getSPredict:
            return {
                's': s_x,
                'k': k_x
            }

        x = self.extract(s_x)
        x_feature = x
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context_1 = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context_1)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output_1 = self.conv2(x)

        pre_context = torch.cat([output_1,x_feature, pre_feature, post_feature], 1)
        x = self.down_conv1_1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x, skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(k_x + output, min=-1, max=1)

        return output

class SAKStackSelfUResnetGeneratorWithoutS(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SAKStackSelfUResnetGeneratorWithoutS, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32+64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_0 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            CBAM(256)
        )

        model = []
        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            CBAM(64)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        # complex conv
        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)

        #fusion
        self.final_conv = nn.Conv2d(2, 1, 1)

    def forward(self, input,pre,post, getKPredict = False, getSPredict = False):
        k_x = torch.fft.fft2(input)
        k_x = torch.fft.fftshift(k_x)
        real = torch.unsqueeze(torch.real(k_x), -1)
        imag = torch.unsqueeze(torch.imag(k_x), -1)
        k_x = torch.cat((real, imag), -1)

        x0 = k_x
        x1 = self.inc(k_x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        k_x = self.up2(x4, x3)
        k_x = self.up3(k_x, x2)
        k_x = self.up4(k_x, x1)
        k_x = self.ouc(k_x)
        k_x = x0 + k_x

        k_real, k_imag = torch.unbind(k_x, dim=-1)
        k_x = torch.complex(k_real, k_imag)
        s_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(k_x)))
        if getKPredict:
            return {
                's': s_x,
                'k': k_x
            }

        x = self.extract(s_x)
        x_feature = x
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context_1 = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context_1)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output_1 = self.conv2(x)

        pre_context = torch.cat([output_1,x_feature, pre_feature, post_feature], 1)
        x = self.down_conv1_1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x, skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(k_x + output, min=-1, max=1)

        return output

class SAKStackSelfUResnetGeneratorWithoutK(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SAKStackSelfUResnetGeneratorWithoutK, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32+64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_0 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            CBAM(256)
        )

        model = []
        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            CBAM(64)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        # complex conv
        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)

        #fusion
        self.final_conv = nn.Conv2d(2, 1, 1)

    def forward(self, input,pre,post, getKPredict = False, getSPredict = False):
        s_x0 = input
        s_x = self.conv1(input)
        s_x = self.down_conv1_0(s_x)
        s_x1 = s_x
        s_x = self.down_conv2(s_x)
        s_x2 = s_x
        s_x = self.res_net(s_x)
        s_x = torch.cat([s_x, s_x2], 1)
        s_x = self.up_conv1(s_x)
        s_x = torch.cat([s_x, s_x1], 1)
        s_x = self.up_conv2(s_x)
        s_x = self.conv2(s_x)
        s_x = s_x0 + s_x
        k_x = torch.fft.fftshift(torch.fft.fft2(s_x))
        if getSPredict:
            return {
                's': s_x,
                'k': k_x
            }

        x = self.extract(s_x)
        x_feature = x
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context_1 = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context_1)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output_1 = self.conv2(x)

        pre_context = torch.cat([output_1,x_feature, pre_feature, post_feature], 1)
        x = self.down_conv1_1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x, skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(k_x + output, min=-1, max=1)

        return output

class SAKStackSelfUResnetGeneratorWithoutStack(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SAKStackSelfUResnetGeneratorWithoutStack, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_0 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            CBAM(256)
        )


        model = []
        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            CBAM(64)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        # complex conv
        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)

        #fusion
        self.final_conv = nn.Conv2d(2, 1, 1)

    def forward(self, input, getKPredict = False, getSPredict = False):
        k_x = torch.fft.fftshift(torch.fft.fft2(input))
        real = torch.unsqueeze(torch.real(k_x), -1)
        imag = torch.unsqueeze(torch.imag(k_x), -1)
        k_x = torch.cat((real, imag), -1)

        x0 = k_x
        x1 = self.inc(k_x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        k_x = self.up2(x4, x3)
        k_x = self.up3(k_x, x2)
        k_x = self.up4(k_x, x1)
        k_x = self.ouc(k_x)
        k_x = x0 + k_x

        k_real, k_imag = torch.unbind(k_x, dim=-1)
        k_x = torch.complex(k_real, k_imag)
        s_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(k_x)))
        if getKPredict:
            return {
                's': s_x,
                'k': k_x
            }

        s_x0 = s_x
        s_x = self.conv1(s_x)
        s_x = self.down_conv1_0(s_x)
        s_x1 = s_x
        s_x = self.down_conv2(s_x)
        s_x2 = s_x
        s_x = self.res_net(s_x)
        s_x = torch.cat([s_x, s_x2], 1)
        s_x = self.up_conv1(s_x)
        s_x = torch.cat([s_x, s_x1], 1)
        s_x = self.up_conv2(s_x)
        s_x = self.conv2(s_x)
        s_x = s_x0 + s_x
        k_x = torch.fft.fftshift(torch.fft.fft2(s_x))
        if getSPredict:
            return {
                's': s_x,
                'k': k_x
            }

        x = self.conv1(s_x)
        x = self.down_conv1(x)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)

        if self.learn_residual:
            # output = input + output
            output = torch.clamp(k_x + output, min=-1, max=1)

        return output

class KStackSelfUResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(KStackSelfUResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32+64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )


        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model = [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        # complex conv
        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)

        #fusion
        self.final_conv = nn.Conv2d(2, 1, 1)

    def forward(self, input,pre,post, getKPredict = False):
        k_x = torch.fft.fft2(input)
        origin_kx = k_x
        real = torch.unsqueeze(torch.real(k_x), -1)
        imag = torch.unsqueeze(torch.imag(k_x), -1)
        k_x = torch.cat((real, imag), -1)

        x = self.extract(input)
        x_feature = x
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context_1 = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context_1)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output_1 = self.conv2(x)

        pre_context = torch.cat([output_1,x_feature, pre_feature, post_feature], 1)
        x = self.down_conv1_1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x, skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)

        x0 = k_x
        x1 = self.inc(k_x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        k_x = self.up2(x4, x3)
        k_x = self.up3(k_x, x2)
        k_x = self.up4(k_x, x1)
        k_x = k_x + x0 if config.unet_global_residual_conn else k_x
        k_x = self.ouc(k_x)

        k_real, k_imag = torch.unbind(k_x, dim= -1)
        k_x = torch.complex(k_real, k_imag)
        if getKPredict:
            return {
                'origin': origin_kx,
                'predict': k_x
            }

        k_x = torch.clamp(torch.real(torch.fft.ifft2(k_x)), min=-1, max=1)

        # fusion mechanism
        content = torch.clamp(self.final_conv(torch.cat((output, k_x), 1)), min=-1 ,max=1)
        return content

class AKStackSelfUResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(AKStackSelfUResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32+64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            CBAM(256)
        )


        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model = [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            CBAM(64)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

        # complex conv
        self.inc = unet_cmplx.InConv(input_nc, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.bottleneck = unet_cmplx.BottleNeck(256, 256, True)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, output_nc)

        #fusion
        self.final_conv = nn.Conv2d(2, 1, 1)

    def forward(self, input,pre,post, getKPredict = False):
        k_x = torch.fft.fft2(input)
        origin_kx = k_x
        real = torch.unsqueeze(torch.real(k_x), -1)
        imag = torch.unsqueeze(torch.imag(k_x), -1)
        k_x = torch.cat((real, imag), -1)

        x = self.extract(input)
        x_feature = x
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context_1 = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context_1)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output_1 = self.conv2(x)

        pre_context = torch.cat([output_1,x_feature, pre_feature, post_feature], 1)
        x = self.down_conv1_1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x, skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)

        x0 = k_x
        x1 = self.inc(k_x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4 = self.bottleneck(x3)
        k_x = self.up2(x4, x3)
        k_x = self.up3(k_x, x2)
        k_x = self.up4(k_x, x1)
        k_x = k_x + x0 if config.unet_global_residual_conn else k_x
        k_x = self.ouc(k_x)

        k_real, k_imag = torch.unbind(k_x, dim= -1)
        k_x = torch.complex(k_real, k_imag)
        if getKPredict:
            return {
                'origin': origin_kx,
                'predict': k_x
            }

        k_x = torch.clamp(torch.real(torch.fft.ifft2(k_x)), min=-1, max=1)

        # fusion mechanism
        content = torch.clamp(self.final_conv(torch.cat((output, k_x), 1)), min=-1 ,max=1)
        return content

class StackSelfUResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(StackSelfUResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding='same', dilation=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32+64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv1_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 32 + 32 + 1, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            CBAM(256)
        )

        model = []
        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            CBAM(128)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            CBAM(64)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )


    def forward(self, input,pre,post):
        x = self.extract(input)
        x_feature = x
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context_1 = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context_1)
        skip1_1 = x
        x = self.down_conv2(x)
        skip2_1 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2_1], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1_1], 1)
        x = self.up_conv2(x)
        output_1 = self.conv2(x)

        pre_context = torch.cat([output_1,x_feature, pre_feature, post_feature], 1)
        x = self.down_conv1_1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x, skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x, skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output

class SelfUResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SelfUResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.extract = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding='same',dilation=1),
            norm_layer(32),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64+32+32, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )


        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model = [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )


    def forward(self, input,pre,post):
        x = self.conv1(input)
        pre=pre.cuda()
        post=post.cuda()
        pre_feature = self.extract(pre)
        post_feature = self.extract(post)
        pre_context = torch.cat([x,pre_feature,post_feature],1)
        x = self.down_conv1(pre_context)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output


class UResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(UResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # 下采样

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )


        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model = [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.res_net = nn.Sequential(*model)

        # 上采样
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )


    def forward(self, input):
        x = self.conv1(input)
        x = self.down_conv1(x)
        skip1 = x
        x = self.down_conv2(x)
        skip2 = x
        x = self.res_net(x)
        x = torch.cat([x,skip2], 1)
        x = self.up_conv1(x)
        x = torch.cat([x,skip1], 1)
        x = self.up_conv2(x)
        output = self.conv2(x)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2

        # 下采样
        # for i in range(n_downsampling): # [0,1]
        # 	mult = 2**i
        #
        # 	model += [
        # 		nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        # 		norm_layer(ngf * mult * 2),
        # 		nn.ReLU(True)
        # 	]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        # 上采样
        # for i in range(n_downsampling):
        # 	mult = 2**(n_downsampling - i)
        #
        # 	model += [
        # 		nn.ConvTranspose2d(
        # 			ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
        # 			padding=1, output_padding=1, bias=use_bias),
        # 		norm_layer(int(ngf * mult / 2)),
        # 		nn.ReLU(True)
        # 	]
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):

	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()

		padAndConv = {
			'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		}

		try:
			blocks = padAndConv[padding_type] + [
				norm_layer(dim),
				nn.ReLU(True)
            ] + [
				nn.Dropout(0.5)
			] if use_dropout else [] + padAndConv[padding_type] + [
				norm_layer(dim)
			]
		except:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		self.conv_block = nn.Sequential(*blocks)

		# self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
		# def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		#     padAndConv = {
		#         'reflect': [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
		#         'replicate': [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
		#         'zero': [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		#     }
		#     try:
		#         blocks = [
		#             padAndConv[padding_type],
		#
		#             norm_layer(dim),
		#             nn.ReLU(True),
		#             nn.Dropout(0.5) if use_dropout else None,
		#
		#             padAndConv[padding_type],
		#
		#             norm_layer(dim)
		#         ]
		#     except:
		#         raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		#
		#     return nn.Sequential(*blocks)

		# blocks = []
		# if padding_type == 'reflect':
		# 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'replicate':
		# 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'zero':
		# 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		# else:
		# 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		#
		# blocks += [
		# 	norm_layer(dim),
		# 	nn.ReLU(True),
		# 	nn.Dropout(0.5) if use_dropout else None
		# ]
		#
		# if padding_type == 'reflect':
		# 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'replicate':
		# 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'zero':
		# 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		# else:
		# 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		#
		# blocks += [
		# 	norm_layer(dim)
		# ]
		#
		# return nn.Sequential(*blocks)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, gpu_ids=[], use_parallel=True, learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
            self, outer_nc, inner_nc, submodule=None,
            outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            model = [
                dModel,
                submodule,
                uModel
            ]
        # model = [
        # 	# Down
        # 	nn.Conv2d( outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        #
        # 	submodule,
        # 	# Up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
        # 	nn.Tanh()
        # ]
        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = [
                dModel,
                uModel
            ]
        # model = [
        # 	# down
        # 	nn.LeakyReLU(0.2, True),
        # 	# up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        # 	norm_layer(outer_nc)
        # ]
        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            model = [
                dModel,
                submodule,
                uModel
            ]
            model += [nn.Dropout(0.5)] if use_dropout else []

        # if use_dropout:
        # 	model = down + [submodule] + up + [nn.Dropout(0.5)]
        # else:
        # 	model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
