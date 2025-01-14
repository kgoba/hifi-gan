import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from fregan2.utils import init_weights, get_padding
from pytorch_wavelets import DWT1DInverse, DWT1DForward

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, h, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.h = h
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        # 1x1 conv for residual connection
        self.conv_pre = nn.ModuleList([
            norm_f(Conv2d(2, 32, (1, 1), (1, 1), padding=(0, 0))),
            norm_f(Conv2d(4, 128, (1, 1), (1, 1), padding=(0, 0))),
            norm_f(Conv2d(8, 512, (1, 1), (1, 1), padding=(0, 0)))

        ])

        # Discriminator
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(32, 128, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(128, 512, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')

    def forward(self, x):
        # DWT and channel-wise concat

        yA, yC= self.dwt(x)
        yAA, yAC = self.dwt(yA)
        yCA, yCC = self.dwt(yC[0])
        yAAA, yAAC = self.dwt(yAA)
        yACA, yACC = self.dwt(yAC[0])
        yCAA, yCAC = self.dwt(yCA)
        yCCA, yCCC = self.dwt(yCC[0])

        x12 = torch.cat((yA, yC[0]), dim=1)
        x6 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)
        x3 = torch.cat((yAAA, yAAC[0], yACA, yACC[0], yCAA, yCAC[0], yCCA, yCCC[0]), dim=1)

        # Reshape
        xes = []
        for xs in [x, x12, x6, x3]:
            b, c, t = xs.shape
            if t % self.period != 0:
                n_pad = self.period - (t % self.period)
                xs = F.pad(xs, (0, n_pad), "reflect")
                t = t + n_pad
            xes.append(xs.view(b, c, t // self.period, self.period))

        x, x12, x6, x3 = xes

        fmap = []
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            if i < 3:
                fmap.append(x)
                # residual connection
                res = self.conv_pre[i](xes[i + 1])
                res = F.leaky_relu(res, LRELU_SLOPE)
                x = (x + res) / torch.sqrt(torch.tensor(2.))
            else:
                fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.h = h

        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, h),
            DiscriminatorP(3, h),
            DiscriminatorP(5, h),
            DiscriminatorP(7, h),
            DiscriminatorP(11, h),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, h, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.h = h
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        # 1x1 convolutions for residual connections
        self.conv_pre0 = nn.ModuleList([
            norm_f(Conv1d(2, 128, 1, 1, padding=0)),
            norm_f(Conv1d(4, 256, 1, 1, padding=0)),
            norm_f(Conv1d(8, 512, 1, 1, padding=0)),
        ])

        self.conv_pre1 = nn.ModuleList([
            norm_f(Conv1d(4, 128, 1, 1, padding=0)),
            norm_f(Conv1d(8, 256, 1, 1, padding=0)),
        ])

        self.conv_pre2 = nn.ModuleList([
            norm_f(Conv1d(8, 128, 1, 1, padding=0)),
        ])

        # CNNs for discriminator
        self.convs0 = self._make_layers(1, norm_f)
        self.convs1 = self._make_layers(2, norm_f)
        self.convs2 = self._make_layers(4, norm_f)

        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

        self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')

    def _make_layers(self, input_channel, norm_f):
        conv_list = nn.ModuleList([
            norm_f(Conv1d(input_channel, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        return conv_list

    def forward(self, x, num_dis):
        if num_dis == 0:
            yA, yC = self.dwt(x)
            yAA, yAC = self.dwt(yA)
            yCA, yCC = self.dwt(yC[0])
            yAAA, yAAC = self.dwt(yAA)
            yACA, yACC = self.dwt(yAC[0])
            yCAA, yCAC = self.dwt(yCA)
            yCCA, yCCC = self.dwt(yCC[0])

            x12 = torch.cat((yA, yC[0]), dim=1)
            x6 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)
            x3 = torch.cat((yAAA, yAAC[0], yACA, yACC[0], yCAA, yCAC[0], yCCA, yCCC[0]), dim=1)

            xes = [x12, x6, x3]
            fmap = []
            for i, l in enumerate(self.convs0):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i in [1, 2, 3]:
                    # residual connection
                    fmap.append(x)
                    res = self.conv_pre0[i - 1](xes[i - 1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
            return x, fmap

        elif num_dis == 1:
            yA, yC = self.dwt(x)
            yAA, yAC = self.dwt(yA)
            yCA, yCC = self.dwt(yC[0])

            x6 = torch.cat((yA, yC[0]), dim=1)
            x3 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)

            xes = [x6, x3]
            fmap = []
            for i, l in enumerate(self.convs1):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i in [1, 2]:
                    # residual connection
                    fmap.append(x)
                    res = self.conv_pre1[i-1](xes[i-1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
            return x, fmap

        else:
            yA, yC = self.dwt(x)
            x3 = torch.cat((yA, yC[0]), dim=1)

            xes = [x3]
            fmap = []
            for i, l in enumerate(self.convs2):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i == 1:
                    # residual connection
                    fmap.append(x)
                    res = self.conv_pre2[i - 1](xes[i-1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)

            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

            return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiScaleDiscriminator, self).__init__()
        self.h = h

        self.discriminators = nn.ModuleList([
            DiscriminatorS(h, use_spectral_norm=True),
            DiscriminatorS(h),
            DiscriminatorS(h),
        ])
        self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # DWT and channel-wise concat
        yA, yC = self.dwt(y)
        yAA, yAC = self.dwt(yA)
        yCA, yCC = self.dwt(yC[0])

        y_down2 = torch.cat((yA, yC[0]), dim=1)
        y_down4 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)

        yA_hat, yC_hat = self.dwt(y_hat)
        yAA_hat, yAC_hat = self.dwt(yA_hat)
        yCA_hat, yCC_hat = self.dwt(yC_hat[0])

        yhat_down2 = torch.cat((yA_hat, yC_hat[0]), dim=1)
        yhat_down4 = torch.cat((yAA_hat, yAC_hat[0], yCA_hat, yCC_hat[0]), dim=1)

        input_dic = {0: [y, y_hat], 1: [y_down2, yhat_down2], 2: [y_down4, yhat_down4]}
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(input_dic[i][0], i)
            y_d_g, fmap_g = d(input_dic[i][1], i)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
