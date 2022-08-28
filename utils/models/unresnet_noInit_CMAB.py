import torch.nn.functional as F
from torch import nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inp = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return inp * self.sigmoid(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        "ECA是指bb2是否加ECA，bb1是本身就不加的"
        super(BasicBlock, self).__init__()
        # bb1
        self.bb1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                 nn.BatchNorm2d(planes))

        # bb2=bb2+ECA
        bb2 = [nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
               nn.BatchNorm2d(planes)]

        # TODO
        if kwargs.get("CA"):
            bb2.append(ChannelAttention(planes))
        if kwargs.get("SA"):
            bb2.append(SpatialAttention())

        self.bb2 = nn.Sequential(*bb2)

        # 旁支下采样
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # x--->-------bb1------bb2------->
        #     ⬇                       ⬆
        #     ---(self.downsample)----
        residual = x

        x = F.relu(self.bb1(x))

        x = self.bb2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = F.relu(x)

        return x


class unresnet_noInit(nn.Module):

    def __init__(self, num_classes=7, CA=[False, True, False, True, False, True, False, True],
                 SA=[False, True, False, True, False, True, False, True]):
        super(unresnet_noInit, self).__init__()

        self.head = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64))

        # __init__(self, inplanes, planes, ECA=False, stride=1, downsample=None, k_size=3):
        self.layer1 = nn.Sequential(BasicBlock(inplanes=64, planes=64, stride=1, CA=CA[0], SA=SA[0]),
                                    BasicBlock(inplanes=64, planes=64, stride=1, CA=CA[1], SA=SA[1]))
        self.layer2 = nn.Sequential(BasicBlock(inplanes=64, planes=128, stride=2, CA=CA[2], SA=SA[2]),
                                    BasicBlock(inplanes=128, planes=128, stride=1, CA=CA[3], SA=SA[3]))
        self.layer3 = nn.Sequential(BasicBlock(inplanes=128, planes=256, stride=2, CA=CA[4], SA=SA[4]),
                                    BasicBlock(inplanes=256, planes=256, stride=1, CA=CA[5], SA=SA[5]))
        self.layer4 = nn.Sequential(BasicBlock(inplanes=256, planes=512, stride=2, CA=CA[6], SA=SA[6]),
                                    BasicBlock(inplanes=512, planes=512, stride=1, CA=CA[7], SA=SA[7]))

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.head(x))  # 64*40*40

        x = self.layer1(x)  # 64*40*40
        x = self.layer2(x)  # 128*20*20
        x = self.layer3(x)  # 256*10*10
        x = self.layer4(x)  # 512*5*5

        x = F.avg_pool2d(x, 4)  # [2, 512, 1, 1]
        x = x.view(x.size(0), -1)  # torch.Size([2, 512])
        x = self.fc(x)

        return x


class unresnet_noInit_CS(nn.Module):
    def __init__(self):
        super(unresnet_noInit_CS, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, True, False, True, False, True, False, True],
                                   SA=[False, True, False, True, False, True, False, True])

    def forward(self, x):
        return self.net(x)

class unresnet_noInit_C1(nn.Module):
    def __init__(self):
        super(unresnet_noInit_C1, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, True,  False, False, False, False, False, False],
                                   SA=[False, False, False, False, False, False, False, False])

    def forward(self, x):
        return self.net(x)


class unresnet_noInit_C2(nn.Module):
    def __init__(self):
        super(unresnet_noInit_C2, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, True, False, False, False, False],
                                   SA=[False, False, False, False, False, False, False, False])

    def forward(self, x):
        return self.net(x)

class unresnet_noInit_C3(nn.Module):
    def __init__(self):
        super(unresnet_noInit_C3, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, False, False, True, False, False],
                                   SA=[False, False, False, False, False, False, False, False])

    def forward(self, x):
        return self.net(x)


class unresnet_noInit_C4(nn.Module):
    def __init__(self):
        super(unresnet_noInit_C4, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, False, False, False, False, True],
                                   SA=[False, False, False, False, False, False, False, False])

    def forward(self, x):
        return self.net(x)

class unresnet_noInit_S1(nn.Module):
    def __init__(self):
        super(unresnet_noInit_S1, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, False, False, False, False, False],
                                   SA=[False, True, False, False, False, False, False, False])

    def forward(self, x):
        return self.net(x)

class unresnet_noInit_S2(nn.Module):
    def __init__(self):
        super(unresnet_noInit_S2, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, False, False, False, False, False],
                                   SA=[False, False, False, True, False, False, False, False])

    def forward(self, x):
        return self.net(x)

class unresnet_noInit_S3(nn.Module):
    def __init__(self):
        super(unresnet_noInit_S3, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, False, False, False, False, False],
                                   SA=[False, False, False, False, False, True, False, False])

    def forward(self, x):
        return self.net(x)

class unresnet_noInit_S4(nn.Module):
    def __init__(self):
        super(unresnet_noInit_S4, self).__init__()
        self.net = unresnet_noInit(num_classes=7,
                                   CA=[False, False,  False, False, False, False, False, False],
                                   SA=[False, False, False, False, False, False, False, True])

    def forward(self, x):
        return self.net(x)
