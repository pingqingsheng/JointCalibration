#!/usr/env/bin python
from typing import Tuple

import torch

from .basecalibrator import BaseCalibrator


class BeliefMatching(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.prior = kwargs['config']['PRIOR']
        self.coef  = kwargs['config']['COEF']
    
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        optimizer.param_groups[0]['lr'] *= 1e-2
        
        return self, optimizer
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        logits = logits[:, :self.num_classes]
        
        alphas = torch.clamp(torch.exp(logits), 0, 100)
        betas  = torch.ones_like(logits)*self.prior
        
        # compute log-likelihood loss: psi(alpha_target) - psi(alpha_zero)
        a_ans  = torch.gather(alphas, -1, targets.unsqueeze(-1)).squeeze(-1)
        a_zero = torch.sum(alphas, -1)
        ll_loss= torch.digamma(a_ans) - torch.digamma(a_zero)        

        # compute kl loss: loss1 + loss2
        #       loss1 = log_gamma(alpha_zero) - \sum_k log_gamma(alpha_zero)
        #       loss2 = sum_k (alpha_k - beta_k) (digamma(alpha_k) - digamma(alpha_zero) )
        loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), -1)
        loss2 = torch.sum( (alphas - betas)*(torch.digamma(alphas) - torch.digamma(a_zero.unsqueeze(-1))), -1)
        kl_loss = loss1 + loss2
        
        loss = ((self.coef*kl_loss - ll_loss)).mean()
        
        return torch.clamp(loss, -10, 10)
        