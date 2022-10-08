#!/usr/env/bin python

from typing import MutableMapping, Tuple, Union

import torch

class BaseCalibrator(torch.nn.Module):
    
    def __init__(self, 
                 calibrate_loader: Union[torch.utils.data.DataLoader, Tuple[torch.utils.data.DataLoader]] =None, 
                 config:MutableMapping=None) -> None:
        super().__init__()
        
        self.calibrate_loader = calibrate_loader
        self.config = config
    
    def forward(self, x:torch.Tensor, mode: str='train'):
        if not hasattr(self, 'model'):
            raise AttributeError('Call calibrate method first!')
        return self.model(x)
    
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer
                      ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        return self, optimizer
    
    def post_calibrate(self, 
                       optimizer: torch.optim.Optimizer) -> None:
        
        if isinstance(self.model, BaseCalibrator):
            self.model.post_calibrate(optimizer)

    @staticmethod
    def criterion(outs: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(outs, labels)