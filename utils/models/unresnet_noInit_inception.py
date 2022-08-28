from torch import nn
import torch
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels=1, pool_features=8):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 16, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self,x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,stride=1,**kwargs):
        "ECA是指bb2是否加ECA，bb1是本身就不加的"
        super(BasicBlock, self).__init__()
        # bb1
        self.bb1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,padding=1, bias=False),
                                 nn.BatchNorm2d(planes))

        # bb2=bb2+ECA
        bb2 = [nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False),
               nn.BatchNorm2d(planes)]

        # TODO
        # if kwargs.get("ECA"):
        #     bb2.append(eca_layer(planes, kwargs.get("k_size")))

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


class unresnet_noInit_inception(nn.Module):

    def __init__(self, num_classes=7):
        super(unresnet_noInit_inception, self).__init__()

        # self.head = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.BatchNorm2d(64))
        self.head=InceptionA()

        #__init__(self, inplanes, planes, ECA=False, stride=1, downsample=None, k_size=3):
        self.layer1 = nn.Sequential(BasicBlock(inplanes=64,  planes=64,  stride=1), BasicBlock(inplanes=64,  planes=64,  stride=1))
        self.layer2 = nn.Sequential(BasicBlock(inplanes=64,  planes=128, stride=2), BasicBlock(inplanes=128, planes=128, stride=1))
        self.layer3 = nn.Sequential(BasicBlock(inplanes=128, planes=256, stride=2), BasicBlock(inplanes=256, planes=256, stride=1))
        self.layer4 = nn.Sequential(BasicBlock(inplanes=256, planes=512, stride=2), BasicBlock(inplanes=512, planes=512, stride=1))

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.head(x))    # 64*40*40

        x = self.layer1(x)  #64*40*40
        x = self.layer2(x)  # 128*20*20
        x = self.layer3(x)  # 256*10*10
        x = self.layer4(x)  # 512*5*5

        x = F.avg_pool2d(x, 4)  # [2, 512, 1, 1]
        x = x.view(x.size(0), -1)   # torch.Size([2, 512])
        x = self.fc(x)

        return x
