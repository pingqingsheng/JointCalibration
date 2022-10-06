#!/usr/env/bin python

from typing import MutableMapping, Tuple

import torch

class BaseCalibrator():
    
    def __init__(self, config:MutableMapping=None) -> None:
        
        self.config = config
        
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer, 
                      calibrateloader: torch.utils.data.DataLoader
                      ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        return model, optimizer
    
    def post_calibrate(self, 
                       model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer, 
                       calibrateloader: torch.utils.data.DataLoader) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        return model, optimizer

    @staticmethod
    def criterion(*args, **kwargs):
        return 0