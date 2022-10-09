#!/usr/env/bin python

import torch
from torch.nn import functional as F

from .basecalibrator import BaseCalibrator


class Focal(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.gamma = kwargs['config']['GAMMA']
        self.size_average = kwargs['config']['SIZE_AVERAGE']  
        
    def get_gamma_list(self, pt):
        
        gamma_list = 3.0*torch.ones(pt.shape[0])
        gamma_list[pt<0.2]  = 5.0
        gamma_list[pt>=0.5] = self.gamma 
        
        return gamma_list
        
    def loss(self, logits:torch.Tensor, target:torch.Tensor):
        
        logits = logits[:, :self.num_classes]
        
        if logits.dim() > 2:
            logits = logits.view(logits.size(0),logits.size(1),-1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1,2)    # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1,logits.size(2))   # N,H*W,C => N*H*W,C            

        logpt = F.log_softmax(logits, dim=1)
        logpt = logpt.gather(1,target.view(-1, 1)).view(-1)
        pt = logpt.exp()
        gammas = self.get_gamma_list(pt).to(logits.device)
        
        loss = (-1 * (1-pt)**gammas * logpt).sum()
        if self.size_average: 
            loss /= len(pt)
            
        return loss
    