'''
contains code for the neural net
inspired by the resnets
'''

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(intermediate_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self.get_layers(architecture[0], intermediate_channels=64, stride=1)
        self.block2 = self.get_layers(architecture[1], intermediate_channels=128, stride=2)
        self.block3 = self.get_layers(architecture[2], intermediate_channels=256, stride=2)
        self.block4 = self.get_layers(architecture[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 128)

    def get_layers(self, num_layers, intermediate_channels, stride):
        layers = []
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels * 4))
        
        layers.append(ConvBlock(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for _ in range(num_layers - 1):
            layers.append(ConvBlock(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    architecture = [1, 1, 1, 1]
    net = Net(architecture)
    inp = torch.randn(1, 3, 224, 224)
    out = net(inp)
    print(out.shape)
