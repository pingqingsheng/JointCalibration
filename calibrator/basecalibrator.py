#!/usr/env/bin python

from typing import MutableMapping, Tuple, Union, List

import torch

class BaseCalibrator(torch.nn.Module):
    
    def __init__(self, 
                 calibrate_loader: Union[torch.utils.data.DataLoader, Tuple[torch.utils.data.DataLoader]] =None, 
                 config:MutableMapping=None) -> None:
        super().__init__()
        
        self.calibrate_loader = calibrate_loader
        self.config = config
        self.num_classes = config['num_classes']
        self.device = config['device']
    
    def forward(self, x:torch.Tensor, **kwargs):
        if not hasattr(self, 'model'):
            raise AttributeError('Call calibrate method first!')
        return self.model(x)
    
    def get_prob(self, logits:torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.softmax(logits[:, :self.num_classes], 1)
    
            
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

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> Union[torch.Tensor, int]:   
        return torch.nn.functional.cross_entropy(logits[:, :self.num_classes], targets)
    
    def criterion(self, logits: Union[torch.Tensor, torch.nn.Module, List], targets: torch.Tensor, **kwargs) -> Union[torch.Tensor, int]:   
        loss = 0
        if isinstance(logits, List):
            for i in range(len(logits)):
                logits_i = logits[i]
                loss +=  self.loss(logits_i, targets)
            loss /= len(logits)
        else:
            logits = logits
            loss = self.loss(logits, targets)
        return loss