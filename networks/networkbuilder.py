#!/usr/env/bin python

import torch

from .networks import ResNet18, ResNet18_MC, ResNet18_GP

class NetWorkBuilder():
    
    def __init__(self, networkname:str, num_classes:int, in_channels:int, **kwargs) -> None:
        
        self.networkname = networkname
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        if 'in_dim' in kwargs:
            self.in_dim = kwargs['in_dim']
        
    def create(self) -> torch.nn.Module:
        
        if self.networkname == 'resnet18':
            self.model = ResNet18(num_classes=self.num_classes, in_channels=self.in_channels)
        
        elif self.networkname == 'resnet18_mc':
            self.model = ResNet18_MC(num_classes=self.num_classes, in_channels=self.in_channels)
            
        elif self.networkname == 'resnet18_gp':
            self.model = ResNet18_GP(in_dim=self.in_dim, in_channels=self.in_channels)
                    