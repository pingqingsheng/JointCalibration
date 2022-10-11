#! /usr/env/bin python

from typing import Tuple, Union, List
import importlib

import torch
from torch.utils.data import BatchSampler, RandomSampler

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
                if hasattr(self.model.model_list[i], 'likelihood'): # TODO: rewrite this part.  After augmenting a GP layer, re-assign common likelihood and gplayer
                    self.model.model_list[i].likelihood = self.model.common_likelihood
                    self.model.model_list[i].model.gp_layer = self.model.common_gplayer
                    # optimizer = self.model.model_list[i].update_optimizer(self.model.model_list[i].model, optimizer, self.model.model_list[i].likelihood)
        else:
            self.model, optimizer = self.augment_network(self.model, optimizer, self.device)

        return self, optimizer

    @staticmethod
    def augment_network(model:torch.nn.Module, optimizer:torch.optim.Optimizer, device:torch.cuda.Device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        network = model
        while isinstance(network, BaseCalibrator):
            network = network.model
        
        if hasattr(model, 'likelihood'): # augment GP module
            in_features, out_features = model.likelihood.num_features, model.likelihood.num_classes
            layer_module   = getattr(importlib.import_module(model.likelihood.__class__.__module__), model.likelihood.__class__.__name__)
            model.likelihood = layer_module(in_features, out_features+1).to(device)
            optimizer = model.update_optimizer(model.model, optimizer, model.likelihood)
        else:
            orig_lr = optimizer.param_groups[0]['lr']
            orig_weightdecay = optimizer.param_groups[0]['weight_decay']
            in_features, out_features = network.linear.in_features, network.linear.out_features
            layer_module   = getattr(importlib.import_module(network.linear.__class__.__module__), network.linear.__class__.__name__)
            network.linear = layer_module(in_features, out_features+1).to(device)
            optimizer.add_param_group({'params':[x for x in network.linear.parameters()], 'lr':orig_lr, 'weight_decay':orig_weightdecay})

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
    

class JointCalibrationV2(JointCalibration):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def pre_calibrate(self, 
                      model: Union[List[torch.nn.Module], torch.nn.Module], 
                      optimizer: torch.optim.Optimizer) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        
        calibrate_sampler = lambda x : BatchSampler(RandomSampler(x, num_samples=int(0.8*x)), batch_size=self.calibrate_loader.batch_size)
        self.calibrate_loader = torch.utils.data.DataLoader(self.calibrate_loader.dataset, sampler=calibrate_sampler)
        
        if hasattr(self.model, 'model_list'): # Ensemble Module
            for i in range(len(self.model.model_list)):
                self.model.model_list[i], optimizer = self.augment_network(self.model.model_list[i], optimizer, self.device)
                if hasattr(self.model.model_list[i], 'likelihood'): # TODO: rewrite this part.  After augmenting a GP layer, re-assign common likelihood and gplayer
                    self.model.model_list[i].likelihood = self.model.common_likelihood
                    self.model.model_list[i].model.gp_layer = self.model.common_gplayer
                    # optimizer = self.model.model_list[i].update_optimizer(self.model.model_list[i].model, optimizer, self.model.model_list[i].likelihood)
        else:
            self.model, optimizer = self.augment_network(self.model, optimizer, self.device)

        return self, optimizer