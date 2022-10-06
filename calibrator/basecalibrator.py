#!/usr/env/bin python

from typing import MutableMapping, Tuple

import torch

class BaseCalibrator(torch.nn.Module):
    
    def __init__(self, config:MutableMapping=None) -> None:
        super().__init__()
        
        self.config = config
    
    def forward(self, x):
        if not hasattr(self, 'model'):
            raise AttributeError('Call calibrate method first!')
        return self.model(x)
    
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer, 
                      calibrateloader: torch.utils.data.DataLoader
                      ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        return self, optimizer
    
    def post_calibrate(self, 
                       optimizer: torch.optim.Optimizer, 
                       calibrateloader: torch.utils.data.DataLoader) -> None:
        
        if isinstance(self.model, BaseCalibrator):
            self.model.post_calibrate(optimizer, calibrateloader)

    @staticmethod
    def criterion(*args, **kwargs):
        return 0