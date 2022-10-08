#!/usr/bin/env python

from typing import Tuple

import torch
import gpytorch

from .basecalibrator import BaseCalibrator


class GP(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.num_data  = len(kwargs['calibrate_loader'].dataset)
        self.num_class = len(torch.unique(torch.tensor(kwargs['calibrate_loader'].dataset.targets)))
        self.encode_dim = kwargs['config']['ENCODE_DIM']
        self.device = kwargs['config']['device']
        
    def forward(self, x:torch.Tensor, **kwargs):
        
        with gpytorch.settings.num_likelihood_samples(32) as _, gpytorch.settings.cholesky_jitter(1e-1) as _:
            z = self.model(x)
            z = z.add_jitter(1e-1).sample_n(32).mean(0)
            logits = z@self.likelihood.mixing_weights.t()  # num_classes x num_data
            
        return logits
    
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
        
        backbone_lr = optimizer.param_groups[0]['lr']
        params_dataptr = [y.data_ptr() for y in optimizer.param_groups[0]['params']]
        gp_hyperparams = [x for x in network.gp_layer.hyperparameters() if x.data_ptr() in params_dataptr]
        gp_viparams    = [x for x in network.gp_layer.variational_parameters() if x.data_ptr() in params_dataptr]
        other_params   = [x for x in optimizer.param_groups[0]['params'] if x.data_ptr() not in [y.data_ptr() for y in gp_hyperparams+gp_viparams]]
        
        optimizer.param_groups[0].update({'params':other_params})
        optimizer.add_param_group({'params':gp_hyperparams, 'lr':backbone_lr*0.01, 'weight_decay':0})
        optimizer.add_param_group({'params':gp_viparams, 'weight_decay':0})
    
        return self, optimizer
    
    def criterion(self, logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        
        model = self.network.gp_layer
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.network.gp_layer, num_data=self.num_data)
        
        with gpytorch.settings.num_likelihood_samples(32) as _, gpytorch.settings.cholesky_jitter(1e-1) as _:
            
            # Get likelihood term and KL term
            num_batch = logits.shape[0]
            log_likelihood = -torch.log(torch.softmax(logits, 1).gather(1, targets.view(-1, 1)).squeeze()).sum().div(num_batch)
            kl_divergence  = model.variational_strategy.kl_divergence().div(self.num_data)

            # Add any additional registered loss terms
            added_loss = torch.zeros_like(log_likelihood)
            for added_loss_term in model.added_loss_terms():
                added_loss.add_(added_loss_term.loss())

            # Log prior term
            log_prior = torch.zeros_like(log_likelihood)
            for _, module, prior, closure, _ in mll.named_priors():
                log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))
            
            return log_likelihood - kl_divergence + log_prior - added_loss
