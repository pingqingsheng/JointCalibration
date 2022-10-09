#!/usr/bin/env python

from typing import Tuple, Union, List
from sklearn.covariance import log_likelihood

import torch
import gpytorch

from .basecalibrator import BaseCalibrator


class GP(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.num_data  = len(kwargs['calibrate_loader'].dataset)
        self.num_class = len(torch.unique(torch.tensor(kwargs['calibrate_loader'].dataset.targets)))
        self.encode_dim = kwargs['config']['ENCODE_DIM']
        
    def forward(self, x:torch.Tensor, mode='train', **kwargs):
        
        with gpytorch.settings.num_likelihood_samples(32) as _, gpytorch.settings.cholesky_jitter(1e-1) as _:
            logits = self.model(x)
        
        return logits
    
    def get_prob(self, logits: Union[List[torch.nn.Module], torch.Tensor], mode:str ='train') -> torch.Tensor:
        with gpytorch.settings.num_likelihood_samples(32) as _, gpytorch.settings.cholesky_jitter(1e-1) as _:
            if isinstance(logits, List):
                pred_prob = 0
                for i in range(len(logits)):
                    logits_i = logits[i]
                    pred_prob += self.likelihood(logits_i).probs.mean(0)
                pred_prob /= len(logits)
            else:
                pred_prob = self.likelihood(logits).probs.mean(0)
        return pred_prob[:, :self.num_class]
    

    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer
                      ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        network = model
        while isinstance(network, BaseCalibrator):
            network = network.model
        self.network = network
        
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=self.encode_dim, num_classes=self.num_class).to(self.device)
        optimizer = self.update_optimizer(self.network, optimizer, self.likelihood)
        
        return self, optimizer
    
    @staticmethod
    def update_optimizer(network:torch.nn.Module, optimizer: torch.optim.Optimizer, likelihood: torch.distributions.Distribution) -> torch.optim.Optimizer:
        
        backbone_lr = optimizer.param_groups[0]['lr']
        orig_params = optimizer.param_groups[0]['params']
        
        gp_hyperparams = [x for x in network.gp_layer.hyperparameters() if x.data_ptr() not in [y.data_ptr() for y in orig_params]]
        gp_viparams    = [x for x in network.gp_layer.variational_parameters() if x.data_ptr() not in [y.data_ptr() for y in orig_params]]
        ll_params      = [x for x in likelihood.parameters() if x.data_ptr() not in [y.data_ptr() for y in orig_params]]
        other_params   = [x for x in network.parameters() if x.data_ptr() not in [y.data_ptr() for y in orig_params+ll_params]]
        
        if len(gp_hyperparams):
            optimizer.add_param_group({'params':gp_hyperparams, 'lr':backbone_lr*0.01, 'weight_decay':0})
        if len(gp_viparams):
            optimizer.add_param_group({'params':gp_viparams, 'weight_decay':0})
        if len(ll_params + other_params):
            optimizer.param_groups[0].update({'params': orig_params+ll_params+other_params})
        
        return optimizer
    
    def loss(self, logits:Union[List[torch.nn.Module], torch.nn.Module], targets:torch.Tensor) -> torch.Tensor:
        with gpytorch.settings.num_likelihood_samples(32) as _, gpytorch.settings.cholesky_jitter(1e-1) as _:
            
            criterion_mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.network.gp_layer, num_data=self.num_data)
            if isinstance(logits, List): # Ensemble
                loss = 0
                for logits_i in logits:
                    loss += -criterion_mll(logits_i, targets)
                loss /= len(logits)
            else:
                loss = -criterion_mll(logits, targets)
            
        return loss