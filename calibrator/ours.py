#! /usr/env/bin python

from typing import Tuple, Union, List
import importlib

import torch

from .basecalibrator import BaseCalibrator


class JointCalibration(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.criterion_mse = torch.nn.MSELoss()
        
    def get_prob(self, logits:Union[torch.Tensor, List[Union[torch.Tensor, torch.distributions.Distribution]]], **kwargs) -> torch.Tensor: 
        
        f = self.model.get_prob(logits)
        
        if isinstance(logits, List): # Ensemble
            g = 0
            for i in range(len(logits)):
                logits_i = logits[i]
                if isinstance(logits_i, torch.Tensor):
                    calibrate_logits = logits_i[:, -1]
                else: # for GP module
                    calibrate_logits = logits_i.sample_n(32).mean(0)[:, -1]
                g += torch.sigmoid(calibrate_logits).view(-1, 1)
            g /= len(logits)
        else:
            if isinstance(logits, torch.Tensor):
                calibrate_logits = logits[:, -1]
            else:
                calibrate_logits = logits.sample_n(32).mean(0)[:, -1]
            g = torch.sigmoid(calibrate_logits).view(-1, 1)
        
        _, pred = f.max(1)
        pred = pred.view(-1, 1)
        f.scatter_(0, pred, g)
        f /= f.sum(1).view(-1, 1)
        
        return f
    
    def pre_calibrate(self, 
                       model: Union[List[torch.nn.Module], torch.nn.Module], 
                       optimizer: torch.optim.Optimizer) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        if hasattr(self.model, 'model_list'): # Ensemble Module
            for i in range(len(self.model.model_list)):
                self.model.model_list[i], optimizer = self.augment_network(self.model.model_list[i], optimizer, self.device)
            if hasattr(self.model.model_list[0], 'likelihood'): # share likelihood
                common_likelihood = self.model.model_list[0].likelihood
                for i in range(len(self.model.model_list)):
                    self.model.model_list[i].likelihood = common_likelihood
        else:
            self.model, optimizer = self.augment_network(self.model, optimizer, self.device)

        return self, optimizer

    @staticmethod
    def augment_network(model:torch.nn.Module, optimizer:torch.optim.Optimizer, device:torch.cuda.Device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        network = model
        while isinstance(network, BaseCalibrator):
            network = network.model
        
        if hasattr(model, 'likelihood'): # for GP module
            in_features, out_features = model.likelihood.num_features, model.likelihood.num_classes
            layer_module   = getattr(importlib.import_module(model.likelihood.__class__.__module__), model.likelihood.__class__.__name__)
            model.likelihood = layer_module(in_features, out_features+1).to(device)
            optimizer = model.update_optimizer(model.model, optimizer, model.likelihood)
        else:
            in_features, out_features = network.linear.in_features, network.linear.out_features
            layer_module   = getattr(importlib.import_module(network.linear.__class__.__module__), network.linear.__class__.__name__)
            network.linear = layer_module(in_features, out_features+1).to(device)
            optimizer.add_param_group({'params':[x for x in network.linear.parameters()]})

        return model, optimizer
        
    def loss(self, logits:Union[torch.Tensor, torch.nn.Module], targets:torch.Tensor) -> Union[torch.Tensor, int]:
        
        if isinstance(logits,  torch.Tensor):
            logits, calibrate_logits = logits[:, :self.num_classes], logits[:, -1]
        else: # for GP Module
            logits = logits.sample_n(32).mean(0)
            logits, calibrate_logits = logits[:, :self.num_classes], logits[:, -1]
        _, pred = logits.max(1)
        correctness = pred.eq(targets).float()
        
        return self.criterion_mse(torch.sigmoid(calibrate_logits).squeeze(), correctness.squeeze())
    
    