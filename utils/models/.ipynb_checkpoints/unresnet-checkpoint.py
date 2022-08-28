import torch
import torch.nn as nn
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,stride=1, k_size=3):
        "ECA是指bb2是否加ECA，bb1是本身就不加的"
        super(BasicBlock, self).__init__()
        # bb1
        self.bb1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,padding=1, bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU(inplace=True))

        # bb2=bb2+ECA
        bb2 = [nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False),
               nn.BatchNorm2d(planes)]

        self.bb2 = nn.Sequential(*bb2)

        # 旁支下采样
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # ReLU(旁支+bb)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.bb1(x)
        x = self.bb2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class unresnet(nn.Module):

    def __init__(self, num_classes=7):
        super(unresnet, self).__init__()

        self.head = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        #__init__(self, inplanes, planes, ECA=False, stride=1, downsample=None, k_size=3):
        self.layer1 = nn.Sequential(BasicBlock(inplanes=64,  planes=64,  stride=1), BasicBlock(inplanes=64,  planes=64,  stride=1))
        self.layer2 = nn.Sequential(BasicBlock(inplanes=64,  planes=128, stride=2), BasicBlock(inplanes=128, planes=128, stride=1))
        self.layer3 = nn.Sequential(BasicBlock(inplanes=128, planes=256, stride=2), BasicBlock(inplanes=256, planes=256, stride=1))
        self.layer4 = nn.Sequential(BasicBlock(inplanes=256, planes=512, stride=2), BasicBlock(inplanes=512, planes=512, stride=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)    # -1,64*10*10

        x = self.layer1(x)  # -1,64*10*10
        x = self.layer2(x)  # -1,128*5*5
        x = self.layer3(x)  # -1,256*3*3
        x = self.layer4(x)  # -1,512*2*2

        x = self.avgpool(x) # -1,512,1,1
        x = torch.flatten(x,1)  # -1,512
        x = self.fc(x)  # -1,7

        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = unresnet().cuda()
    summary(model, input_size=(1, 40, 40))
    print(1)
