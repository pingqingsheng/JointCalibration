#!/usr/env/bin python
from typing import Tuple

import torch

from .basecalibrator import BaseCalibrator

class CSKD(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.temperature = kwargs['config']['TEMPERATURE']
        self.lamda = kwargs['config']['LAMBDA']

    def loss(self, logits:torch.Tensor, labels:torch.Tensor, **kwargs):
        
        logits = logits[:, :self.num_classes]
        
        n_images = logits.shape[0]
        labels = labels[:n_images//2]
        logits, logits_tilde = logits[:n_images//2, :self.num_classes], logits[(n_images//2):, :self.num_classes]
        
        return torch.nn.functional.cross_entropy(logits, labels) + self.lamda*self.kdloss(logits, logits_tilde)
    
    @staticmethod
    def kdloss(inputs: torch.Tensor, targets: torch.Tensor, temperature: float=1):
        log_p = torch.log_softmax(inputs/temperature, dim=1)
        q = torch.softmax(targets/temperature, dim=1)
        return torch.nn.functional.kl_div(log_p, q)*(temperature**2)/inputs.size(0)