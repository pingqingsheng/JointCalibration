from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import gpytorch

import pickle as pkl

# >>> Original ResNet18 Basic Block
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# >>> MCDrop ResNet18 Basic Block
class BasicBlock_MC(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_p=0.5):
        super(BasicBlock_MC, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout1 = nn.Dropout(drop_p)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout2 = nn.Dropout(drop_p)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout2(F.relu(out))
        return out

# ResNet18 Original
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def partial_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out
    
    def forward(self, x):
        out = self.partial_forward(x)
        out = self.linear(out)
        return out

# ResNet18 MCDrop
class ResNet_MC(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels, drop_p=0.5):
        super(ResNet_MC, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(drop_p)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def partial_forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out
    
    def forward(self, x):
        out = self.partial_forward(x)
        out = self.linear(out)
        return out

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
def ResNet18(num_classes:int=10, in_channels:int=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def ResNet34(num_classes:int=10, in_channels:int=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def ResNet18_MC(num_classes:int=10, in_channels:int=3):
    return ResNet_MC(BasicBlock_MC, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

def ResNet34_MC(num_classes:int=10, in_channels:int=3):
    return ResNet_MC(BasicBlock_MC, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def test():
    net = ResNet18(num_classes=10, in_channels=3)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    
    net = ResNet18_MC(num_classes=10, in_channels=3)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size)


# >>> ResNet18 for GP
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=128):
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, 
            batch_shape=torch.Size([num_dim])
        )
        
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, 
                grid_size = grid_size, 
                grid_bounds=[grid_bounds], 
                variational_distribution=variational_distribution
            ), num_tasks=num_dim
        )
        
        super().__init__(variational_strategy)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    
class ResNet18_GP(nn.Module):
    
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    
    def __init__(self, in_dim, in_channels, grid_bounds=(-10. , 10.), pretrained=False, **kwargs) -> None:
        super().__init__()
        self.encoder = ResNet18(num_classes=in_dim, in_channels=in_channels)
        self.gp_layer = GaussianProcessLayer(num_dim=self.encoder.linear.out_features, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = self.encoder.linear.out_features
        self.scaling = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])
        
        if pretrained:
            self.encoder.load_state_dict(model_zoo.load_url(self.model_urls['resnet18']))
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.scaling(x).transpose(-1, -2).unsqueeze(-1)
        x = self.gp_layer(x)
        return x