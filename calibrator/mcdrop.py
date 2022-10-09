#!/usr/env/bin python
from typing import Tuple, Union

import torch

from .basecalibrator import BaseCalibrator


class MCDrop(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.num_samples  = int(kwargs['config']['NUM_SAMPLES'])
        self.dropout_prob = float(kwargs['config']['DROPOUT_PROB']) 
        
    def forward(self, x:torch.Tensor, mode:str='train'):
        
        if mode == 'train':
            outs = self.model(x)
        elif mode == 'eval':
            b, c, w, h = x.shape
            x_aug = x.unsqueeze(1).repeat(1, self.num_samples, 1, 1, 1)
            x_aug = x_aug.reshape(-1, c, w, h)
            n_aug = x_aug.shape[0]
            n_iter = n_aug // len(x)
            
            dropout_outs = []
            for ib in range(int(n_iter)):
                outs = self.model(x_aug[ib*len(x):min(n_aug, (ib+1)*len(x))])
                dropout_outs.append(outs.squeeze())
            
            outs = torch.cat(dropout_outs, 0).reshape(len(x), self.num_samples, -1).mean(1)

        return outs
    
    def _pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer, 
                      **kwargs) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        network = self.model
        while isinstance(network, BaseCalibrator):
            network = network.model 
        network.enable_dropout()
        
        return self, optimizer
    
    @staticmethod
    def loss(*args, **kwargs)-> Union[torch.Tensor, int]:
        return 0