#!/usr/env/bin python
from typing import Tuple

import torch
from copy import deepcopy

from .basecalibrator import BaseCalibrator

class Ensemble(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.num_models = kwargs['config']['NUM_MODELS']
        self.device = kwargs['config']['device']
    
    def forward(self, x):
        
        ensemble_outs = 0
        for model in self.model_list: 
            ensemble_outs += model(x)
        
        return ensemble_outs/len(self.model_list)
    
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer, 
                      **kwargs) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        self.model_list = []
        for i in range(self.num_models):
            
            model_i = deepcopy(model)
            for module in model_i.children():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
                    
            self.model_list.append(model_i.to(self.device))
        
            optimizer.add_param_group({'params':model_i.parameters()})

        return self, optimizer
    
    @staticmethod
    def criterion(*args, **kwargs):
        return 0