import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_rate=None, act='l'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.bn = nn.BatchNorm2d(out_channels)

        if act == 'm':
            # self.act = Mish()
            pass
        elif act == 'l':
            self.act = nn.LeakyReLU(0.1)
        elif act == 'r':
            self.act = nn.ReLU()
        elif act == 's':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

        if dropout_rate is not None and dropout_rate != 0.:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_rate=None, act='l'):
        super(SeparableConvBlock, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=kernel_size // 2, groups=in_channels, bias=False)
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

        self.bn = nn.BatchNorm2d(out_channels)

        if act == 'm':
            # self.act = Mish()
            pass
        elif act == 'l':
            self.act = nn.LeakyReLU(0.2)
        elif act == 'r':
            self.act = nn.ReLU()
        elif act == 's':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

        if dropout_rate is not None and dropout_rate != 0.:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, name, fusion_type='add'):
        super(FusionBlock, self).__init__()

        if fusion_type == 'add':
            self.fusion = nn.Sequential(nn.Identity())

        elif fusion_type == 'max':
            self.fusion = nn.Sequential(nn.MaxPool2d(kernel_size=1, stride=1))

        elif fusion_type == 'con':
            self.fusion = nn.Sequential(nn.Identity())

        elif fusion_type == 'avg':
            self.fusion = nn.Sequential(nn.AvgPool2d(kernel_size=1, stride=1))

        self.name = name

    def forward(self, *tensors):
        if len(tensors) == 1:
            return tensors[0]
        else:
            if self.fusion[0] is nn.Identity():
                return nn.torch.add(*tensors, name='add_' + self.name)
            elif self.fusion[0] is nn.MaxPool2d(kernel_size=1, stride=1):
                return nn.torch.max(nn.torch.stack(tensors), dim=0, keepdim=False, name='max_' + self.name)[0]
            elif self.fusion[0] is nn.AvgPool2d(kernel_size=1, stride=1):
                return nn.torch.mean(nn.torch.stack(tensors), dim=0, keepdim=False, name='avg_' + self.name)
            elif self.fusion[0] is nn.Identity():
                return nn.torch.cat(tensors, dim=1, name='conc_' + self.name)


class AtrousBlock(nn.Module):
    def __init__(self, in_channels, ind=0, nf=32, fs=3, strides=1, act='l', dropout_rate=None, weight_decay=5e-4,
                 pool=0, FUS='max'):
        super(AtrousBlock, self).__init__()

        self.ind = ind
        self.nf = nf
        self.fs = fs
        self.strides = strides
        self.act = act
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.pool = pool
        self.FUS = FUS

        self.ki = nn.init.kaiming_normal_
        self.kr = nn.regularizers.l2(weight_decay)
        self.x = []
        self.d = []
        self.ab = 3

        self.redu_r = in_channels // 2
        if ind > 0:
            self.redu_conv = nn.Conv2d(in_channels, self.redu_r, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_redu = nn.BatchNorm2d(self.redu_r)

        self.depthwise_convs = nn.ModuleList()
        self.bn_depthwise = nn.ModuleList()

        for i in range(self.ab):
            self.depthwise_convs.append(
                nn.Conv2d(self.redu_r, self.redu_r, kernel_size=self.fs, stride=self.strides, padding=self.fs // 2,
                          dilation=i+1, groups=self.redu_r, bias=False))
            self.bn_depthwise.append(nn.BatchNorm2d(self.redu_r))

        self.fusion_block = FusionBlock(in_channels)

        self.conv = nn.Conv2d(in_channels, nf, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ind > 0:
            x_i = self.redu_conv(x)
            x_i = self.bn_redu(x_i)
            x_i = F.relu(x_i)
        else:
            x_i = x

        for i in range(self.ab):
            self.x.append(self.mininet(x_i, i, self.ind))
            self.d.append(self.x[i].shape[1])

        mr = [x_i]

        for i in range(0, len(self.d)):
            if self.d[0] == self.d[i]:
                mr.append(self.x[i])

        if len(mr) > 1:
            f = self.fusion_block(*mr)
        else:
            f = self.x[0]

        b = self.conv(f)
        b = self.bn(b)
        b = F.relu(b)

        if self.dropout_rate is not None and self.dropout_rate != 0.:
            b = nn.Dropout(self.dropout_rate)(b)

        return b

    def mininet(self, x, dr, ind):
        m = self.depthwise_convs[dr](x)
        m = self.bn_depthwise[dr](m)
        m = F.relu(m)
        # if dropout_rate != None and dropout_rate != 0.:
        #     m = Dropout(dropout_rate)(m)
        return m

