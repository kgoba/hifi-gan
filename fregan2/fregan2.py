import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from fregan2.utils import init_weights, get_padding
from pytorch_wavelets import DWT1DInverse, DWT1DForward

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size, dilation):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class Generator(torch.nn.Module):
    def __init__(self, hparams):
        super(Generator, self).__init__()
        self.hparams = hparams
        self.num_kernels = len(hparams.resblock_kernel_sizes)
        self.num_upsamples = len(hparams.upsample_rates)

        self.conv_pre = Conv1d(80, hparams.upsample_initial_channel, 7, 1, padding=3)

        resblock = ResBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hparams.upsample_rates, hparams.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hparams.upsample_initial_channel // (2 ** (i)),
                                hparams.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = hparams.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(hparams.resblock_kernel_sizes, hparams.resblock_dilation_sizes)):
                self.resblocks.append(resblock(hparams, ch, k, d))

        self.conv_post= Conv1d(ch, 2, 7, 1, padding=3, bias=False)

        self.ups.apply(init_weights)
        self.idwt = DWT1DInverse()

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)

        x = self.conv_post(x)
        x = torch.tanh(x)

        x_low, x_high = x.chunk(2, dim=1)

        x = self.idwt([x_low, [x_high]])

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class Generator2(torch.nn.Module):
    def __init__(self, hparams):
        super(Generator2, self).__init__()
        self.hparams = hparams
        self.num_kernels = len(hparams.resblock_kernel_sizes)
        self.num_upsamples = len(hparams.upsample_rates)

        self.conv_pre = Conv1d(80, hparams.upsample_initial_channel, 7, 1, padding=3)

        resblock = ResBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hparams.upsample_rates, hparams.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hparams.upsample_initial_channel // (2 ** (i)),
                                hparams.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = hparams.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(hparams.resblock_kernel_sizes, hparams.resblock_dilation_sizes)):
                self.resblocks.append(resblock(hparams, ch, k, d))

        self.conv_post= Conv1d(ch, 4, 7, 1, padding=3, bias=False)

        self.ups.apply(init_weights)
        self.idwt = DWT1DInverse()

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)

        x = self.conv_post(x)
        x = torch.tanh(x)

        x_low_low, x_low_high, x_high_low, x_high_high = x.chunk(4, dim=1)

        x_low = self.idwt([x_low_low, [x_low_high]])
        x_high = self.idwt([x_high_low, [x_high_high]])
        x = self.idwt([x_low, [x_high]])

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)

    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l

    return loss

