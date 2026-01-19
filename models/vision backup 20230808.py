import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms


# def weights_init(m):
#     if hasattr(m, "weight"):
#         m.weight.data.uniform_(-0.5, 0.5)
#     if hasattr(m, "bias"):
#         m.bias.data.uniform_(-0.5, 0.5)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        act = nn.Sigmoid

        self.fc = nn.Sequential(
            nn.Linear(3072, 1000), 
            act(), 
            nn.Linear(1000, 500), 
            act(),
            nn.Linear(500, 100), 
        )
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
    
class FCN1(nn.Module):  # Only 1 hidden layer
    def __init__(self):
        super(FCN1, self).__init__()
        # act = nn.Sigmoid

        self.fc = nn.Sequential(
            nn.Linear(3072, 100), 
        )
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
    
class CFN(nn.Module):  # Only 1 Conv + 1 hidden layer
    def __init__(self):
        super(CFN, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4096, 100), 
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
    
        
class CCNN(nn.Module):
    def __init__(self):
        super(CCNN, self).__init__()
        act1 = nn.Sigmoid
        self.sigmoid = torch.nn.Sigmoid()
        # act2 = nn.LeakyReLU
        self.body0 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5, padding=5//2, stride=1),
            act1(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, padding=5//2, stride=1),
            act1(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4096, 100),

        )
        
    def forward(self, x):
        out = self.body0(x)
        out = self.body(out)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
    

class CCNN_support(nn.Module):
    def __init__(self):
        super(CCNN_support, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        act1 = nn.Sigmoid
        # act2 = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, padding=5//2, stride=1),
            act1(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4096, 100),

        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

class LeNet(nn.Module):
    def __init__(self, num_classes=100, n_h=768):   #n_h: Number of hidden units
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),)
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),)
        self.fc = nn.Linear(n_h, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         act = nn.Sigmoid
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 4, kernel_size=5, padding=5//2, stride=1),
#             act(),)
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(4, 4, kernel_size=5, padding=5//2, stride=1),
#         #     act(),)
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(4, 4, kernel_size=5, padding=5//2, stride=1),
#             act(),)
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(4, 4, kernel_size=5, padding=5//2, stride=1),
#             act(),)
#         self.fc = nn.Linear(4096, 100)
        
#     def forward(self, x):
#         out = self.conv1(x)
#         # out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out




'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

value = 0.02
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-value, value)
    if hasattr(m, "bias") and m.bias != None:
        m.bias.data.uniform_(-value, value)



    # if hasattr(m, "weight"):
    #     m.weight.data.uniform_(-value, value)
    # if hasattr(m, "bias") and m.bias != None:
    #     m.bias.data.uniform_(-value, value)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = F.sigmoid(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, n_h = 24*64):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 4 conv layers
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)  
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)  
        # print(block.expansion, num_classes)
        # self.linear = nn.Linear(512*64, num_classes)

        # Make ResNet is smaller
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)   # 4 conv layers
        self.layer2 = self._make_layer(block, 12, num_blocks[1], stride=1)  
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 24, num_blocks[3], stride=1)  
        self.linear = nn.Linear(n_h, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.sigmoid(self.bn1(self.conv1(x)))
        out = F.sigmoid(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size = 4, stride = 4)
        out = out.view(out.size(0), -1)   # the size -1 is inferred from other dimensions
        out = self.linear(out)
        return out


class MyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(MyResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Make ResNet is smaller
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)   # 4 conv layers
        self.linear = nn.Linear(16*16, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.avg_pool2d(out, kernel_size = 4, stride = 4)
        out = out.view(out.size(0), -1)   # the size -1 is inferred from other dimensions
        out = self.linear(out)
        return out
    
from math import ceil
############## Implement EfficientNet

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    "b0": (0, 224, 0.2),   # alpha, beta, gamma, depth = alpha**phi
    "b1": (0.5, 240, 0.2),
}


class CNNBlock(nn.Module):
    def __intit__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride, 
            padding, 
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()


    


class SqueezeExcitation(nn.Module):
    def __init__(self) -> None:
        def __init__(self, in_channels, reduced_dim):
            super(SqueezeExcitation, self).__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), # C x H x W --> C x 1 x 1
                nn.Conv2d(in_channels, reduced_dim, 1),
                nn.SiLU(),
                nn.Conv2d(reduced_dim),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return x*self.se(x)
    


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4,
            survival_prob=0.8,):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,)
        
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
    
    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob

        return torch.div(x, self.survival_prob)*binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs




class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280*width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32*width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels
    

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*width_factor)/4)
            layers_repeats = ceil(repeats*depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                    in_channels,
                    out_channels,
                    expand_ratio=expand_ratio,
                    stride= stride if layer==0 else 1,
                    kernel_size=kernel_size,
                    padding=kernel_size//2, # if k=1:pad=0, k=3:pad=1, k=5:pad = 2
                    )
                )

                in_channels = out_channels
        
        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1,padding=0)
        )

        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def EffNet():
    return EfficientNet(version=0, num_classes=100)





def ResNet18(n_classes=100, n_h=24*64):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=n_classes, n_h=n_h)

def ResNet5():
    return MyResNet(BasicBlock, [2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

