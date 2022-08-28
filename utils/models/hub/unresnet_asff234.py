from utils.models.unresnet_noInit import unresnet_noInit
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    相当于是一个block，只需要输入conv信息，自动形成conv-BN-activate function
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        # 输入的三个特征层的channels, 根据实际修改
        self.dim = [512,256, 128]
        self.inter_dim = self.dim[self.level]
        # 每个层级三者输出通道数需要一致
        if level == 0:
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)   # 128-->256
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)   #  64-->256
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)                  # 256-->1024
        elif level == 1:  #
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    # 尺度大小 level_0(256,5,5) < level_1(128,10,10) < level_2(64,20,20)
    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0: # 256
            level_0_resized = x_level_0 # 256,5,5
            level_1_resized = self.stride_level_1(x_level_1)    # 128,10,10 --> 256,5,5

            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1) # 64,20,20 --> 64,10,10
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)    # 64,20,20 --> 256,5,5

        elif self.level == 1:   # 128
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:   # 64
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            else:
                level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)  # alpha等产生

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]
        out=fused_out_reduced
        # out = self.expand(fused_out_reduced)    # 256,5,5 --> 1024,5,5

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class unresnet_ASFF234(unresnet_noInit):
    def __init__(self):
        super(unresnet_ASFF234, self).__init__()
        self.asff = ASFF(level=0)
    def forward(self, x):
        h1 = F.relu(self.head(x))

        out1 = self.layer1(h1)  # 64*40*40
        out2 = self.layer2(out1)  # 128*20*20
        out3 = self.layer3(out2)  # 256*10*10
        out4 = self.layer4(out3)  # 512*5*5

        out4_asff = self.asff(out4, out3, out2)

        x = F.avg_pool2d(out4_asff, 4)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = unresnet_ASFF234().cuda()
    summary(model, input_size=(1, 40, 40))
    print(1)

