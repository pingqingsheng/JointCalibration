#!/usr/env/bin python
from typing import Tuple

import torch

from .basecalibrator import BaseCalibrator

class CSKD(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.temperature = kwargs['config']['TEMPERATURE']
        self.lamda = kwargs['config']['LAMBDA']

        self.criterion_ce = torch.nn.CrossEntropyLoss()
        
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer, 
                      **kwargs) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        return self, optimizer
    
    def criterion(self, outs, labels, **kwargs):
        
        n_images = outs.shape[0]
        labels = labels[:n_images//2]
        outs, outs_tilde = outs[:n_images//2], outs[(n_images//2):]
        
        return self.criterion_ce(outs, labels) + self.kdloss(outs, outs_tilde)
    
    @staticmethod
    def kdloss(inputs: torch.Tensor, targets: torch.Tensor, temperature: float=1):
        log_p = torch.log_softmax(inputs/temperature, dim=1)
        q = torch.softmax(targets/temperature, dim=1)
        return torch.nn.functional.kl_div(log_p, q)*(temperature**2)/inputs.size(0)