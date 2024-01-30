from typing import List

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
            self.act = Mish()
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


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    @staticmethod
    def forward(x):
        return x * (torch.tanh(F.softplus(x)))


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_rate=None, act='l'):
        super(SeparableConvBlock, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding="same", groups=in_channels, bias=False)
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

        self.bn = nn.BatchNorm2d(out_channels)

        if act == 'm':
            self.act = Mish()
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
        self.name = name
        self.type = fusion_type

    def forward(self, tensors: List[torch.Tensor]):
        if self.type == 'add':
            out = tensors[0]
            for tensor in tensors[1:]:
                out = torch.add(out, tensor)
            return out

        # if self.type == 'max':
        #     out = torch.max(*tensors)
        #     out_named = out.clone().detach().requires_grad_(True)
        #     out_named.set_(out_named, name='max_' + self.name)
        #     return out_named

        # if self.type == 'con':
        #     out = torch.cat(tensors, dim=1)
        #     out_named = out.clone().detach().requires_grad_(True)
        #     out_named.set_(out_named, name='con_' + self.name)
        #     return out_named

        # if self.type == 'avg':
        #     out = torch.mean(torch.stack(tensors, dim=0), dim=0)
        #     out_named = out.clone().detach().requires_grad_(True)
        #     out_named.set_(out_named, name='avg_' + self.name)
        #     return out_named


class Activation(nn.Module):
    def __init__(self, name, t='-', n=255):
        super(Activation, self).__init__()
        self.name = name
        self.t = t
        self.n = n

    def forward(self, x):
        if self.t == 'r':
            return F.relu(x, inplace=True)
        if self.t == 'l':
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)
        if self.t == 'e':
            return F.elu(x, inplace=True)
        if self.t == 'n':
            return torch.clamp(F.relu(x), 0, self.n)
        if self.t == 'hs':
            return F.hardsigmoid(x)
        if self.t == 's':
            return torch.sigmoid(x)
        if self.t == 't':
            return torch.tanh(x)
        # if self.t == 'm':
        #     return Mish.forward(x)
        return x


class AtrousBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=3, stride=1, act='l', dropout_rate=None,
                 fusion_type='add'):
        super(AtrousBlock, self).__init__()
        self.redu_r = 16
        self.kernel_size = kernel_size
        self.stride = stride
        self.act = act
        self.dropout_rate = dropout_rate
        self.fusion_type = fusion_type

        self.conv_redu = nn.Conv2d(in_channels, self.redu_r, kernel_size=1, stride=1, padding='same', bias=False)
        nn.init.kaiming_normal_(self.conv_redu.weight, mode='fan_out', nonlinearity='relu')
        self.bn_redu = nn.BatchNorm2d(self.redu_r)
        self.act_redu = Activation(name='act_redu', t=self.act)

        self.conv_depth_list = nn.ModuleList([
            nn.Conv2d(self.redu_r, self.redu_r, kernel_size=kernel_size, stride=stride,
                      padding='same', dilation=(i+1), groups=self.redu_r, bias=False) for i in range(3)])
        for conv_depth in self.conv_depth_list:
            nn.init.kaiming_normal_(conv_depth.weight, mode='fan_out', nonlinearity='relu')

        self.bn_list = nn.ModuleList([nn.BatchNorm2d(self.redu_r) for _ in range(3)])
        self.act_depth = Activation(name='atrous_act', t=self.act)

        self.conv_out = nn.Conv2d(self.redu_r, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.conv_out.weight, mode='fan_out', nonlinearity='relu')
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act_out = Activation(name='act_out', t=self.act)

        self.fusion_block = FusionBlock(name='atrous_Fusion_block', fusion_type=self.fusion_type)

    def forward(self, x):
        self.redu_r = x.size(1) // 2   # reduction ratio -> get channels - NCHW - channels first format
        x_redu = self.conv_redu(x)
        x_redu = self.bn_redu(x_redu)
        x_redu = self.act_redu(x_redu)

        # x_depth_list = [self.conv_depth_list[i](x_redu) for i in enumerate(depth3)]
        x_depth_list = [self.conv_depth_list[0](x_redu), self.conv_depth_list[1](x_redu), self.conv_depth_list[2](x_redu)]
        x_depth_list = [self.bn_list[0](x_depth_list[0]), self.bn_list[1](x_depth_list[1]),
                        self.bn_list[2](x_depth_list[2])]
        x_depth_list = [self.act_depth(x_depth_list[0]), self.act_depth(x_depth_list[1]), self.act_depth(x_depth_list[2])]

        # x_depth_list = [F.interpolate(x_depth_list[i], size=x_depth_list[0].shape[2:], mode='bilinear',
        #                               align_corners=False) for i in range(3)]
        x_depth_list = [F.interpolate(x_depth_list[0], size=x_depth_list[0].shape[2:], mode='bilinear', align_corners=False),
                        F.interpolate(x_depth_list[1], size=x_depth_list[0].shape[2:], mode='bilinear', align_corners=False),
                        F.interpolate(x_depth_list[2], size=x_depth_list[0].shape[2:], mode='bilinear', align_corners=False)]

        x_fused = self.fusion_block(x_depth_list + [x_redu])

        x_out = self.conv_out(x_fused)
        x_out = self.bn_out(x_out)
        x_out = self.act_out(x_out)

        if self.dropout_rate is not None and self.dropout_rate != 0.:
            x_out = nn.Dropout2d(p=self.dropout_rate)(x_out)

        return x_out


class ACFFModel(nn.Module):
    def __init__(self, num_classes):
        super(ACFFModel, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.act0 = nn.ReLU(inplace=True)
        self.block1 = AtrousBlock(32, 64, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = AtrousBlock(64, 96, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = AtrousBlock(96, 128, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = AtrousBlock(128, 128, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = AtrousBlock(128, 128, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.block6 = AtrousBlock(128, 128, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.block7 = AtrousBlock(128, 256, kernel_size=3, stride=1, act='l', dropout_rate=None, fusion_type='add')
        self.sepconv = SeparableConvBlock(256, num_classes, kernel_size=1, stride=1, dropout_rate=None, act='l')
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))  # Global Average pool
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.sepconv(x)
        x = self.pool5(x)  # Global Average pool
        x = x.view(x.size(0), -1)  # Flatten the output
        cls = self.softmax(x)

        return cls
